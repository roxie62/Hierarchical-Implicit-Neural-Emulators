import argparse
import datetime
import numpy as np
import os
import time,pdb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_mar import train_one_epoch, eval_model_rec, roll_out_seq
import copy
from dataloader import DistributedPreloadDataset

def get_args_parser():
    parser = argparse.ArgumentParser('FutureINN', add_help=False)

    # Model parameters
    parser.add_argument('--ds_factor', default=4, type=int)
    parser.add_argument('--ds_factor_low', default=16, type=int)
    parser.add_argument('--ratio', default=0.85, type=float)
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--num_of_frame', type=int, default=3)

    # Training and evaluation parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--eval_freq', type=int, default=25, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=50, help='save last frequency')
    parser.add_argument('--evaluate', action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--acc_grad', type=int, default=1, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9995, type=float)
    parser.add_argument('--eval_idx', type=int, default=0)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_epoch', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    log_writer = None

    ######################################### dataset ##############################################
    dataset_train = DistributedPreloadDataset(args.num_of_frame, args.img_size)
    dataset_validation = DistributedPreloadDataset(args.num_of_frame, args.img_size, validation = True)
    dataset_test = DistributedPreloadDataset(args.num_of_frame, args.img_size, eval = True)
    dataset_train[0]
    sampler_train = dataset_train
    print('train dataset', len(dataset_train))
    print('validation dataset', len(dataset_train))
    print('test dataset', len(dataset_test), '\n')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle = True,
    )

    from model import FutureINN 
    model = FutureINN(args)

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    ################################## distributed training and optimizer ##################################
    eff_batch_size = args.batch_size * misc.get_world_size()
    print("base lr: %.2e" % (args.lr / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters = True)
        model_without_ddp = model.module
    model_without_ddp.ema_step(0) 
    
    params = []
    for name, param in model_without_ddp.named_parameters():
        params.append(param)
    optimizer = torch.optim.AdamW([{'params':[], 'lr':args.lr}, {'params':params, 'lr': args.lr, 'lr_scale':1.0}], \
                            weight_decay=0.03, betas = (0.99, 0.999))
    loss_scaler = NativeScaler()

    ########################################### resume training ##############################################
    ckpt_pth = "checkpoint-last.pth" if args.resume_epoch < 1 else f"checkpoint-{args.resume_epoch}.pth"
    if args.resume and os.path.exists(os.path.join(args.resume, ckpt_pth)):
        checkpoint = torch.load(os.path.join(args.resume, ckpt_pth), map_location='cpu', weights_only = False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    if args.gpu == 0 and args.evaluate:
        import pandas as pd
        from tqdm import tqdm
        def save_folder(prefix, args):
            pth = '/'.join(os.getcwd().split('/')[:-1]) + f'/{prefix[1]}'
            os.makedirs(pth, exist_ok=True)
            pth = pth + f'/{prefix[0]}_{args.ds_factor}_{args.ds_factor_low}_{args.start_epoch}' 
            os.makedirs(pth, exist_ok = True)
            return pth

        save_pth = save_folder(['ours_level3', 'mse'], args)
        save_data_pth = save_folder(['ours_level3', 'data'], args)
        save_data_pth_long = save_folder(['ours_level3', 'data_long'], args)

        torch.cuda.empty_cache()
        prefix = 'ours_level3'
        mse_list, mse_save = [], []

        if False: ## this is for generating a long long sequence
            tms = 200000 
            os.makedirs('imgs', exist_ok=True)
            save_folder = save_data_pth_long + f'/{args.eval_idx}'
            os.makedirs(save_folder, exist_ok=True)
            eval_model_rec(model, dataset_test, btz=1, sidx=args.eval_idx, output_dir=args.output_dir, tms=tms, save_folder = save_folder)
            exit()

        mse_list, mse_save = [], []
        for sidx in tqdm(range(0, 2500, 25)):
            model.eval()
            data = eval_model_rec(model, dataset_test, args.start_epoch, btz=1, sidx=sidx, output_dir=args.output_dir, tms=200)
            mse_list.append(data['mse'])
            torch.save(data, save_data_pth+f'/{sidx}.pth')

        mse_array = torch.stack(mse_list, dim=0)
        mse_mean = mse_array.mean(dim=0)
        mse_std = mse_array.std(dim=0)
        for i in (0, 5, 10, 20, 30, 40, 50, -1):
            print(i, f"mse mean: {mse_mean[i]:.5f}, mse std: {mse_std[i]:.5f}")
            mse_save.append([i, mse_mean[i].cpu().data.numpy(), mse_std[i].cpu().data.numpy()])
        df = pd.DataFrame(dict(time=np.stack(mse_save)[:,0], mse=np.stack(mse_save)[:,1], mse_std=np.stack(mse_save)[:,2]))
        df.to_csv(f'{args.output_dir}/csv/eval_mse_{args.start_epoch:04d}.csv', index=False, float_format='%.5f')
        df.to_csv(save_pth+'.csv', index=False, float_format='%.5f')

    ################################################## training ##################################################
    if not args.evaluate:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            train_one_epoch(
                model, 
                model_params, ema_params,
                data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                dataset_validation=dataset_validation,
            )

            # save checkpoint
            if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")
            if epoch > 100 and epoch % args.save_last_freq == 0:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name=f"{epoch}")

            if misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
