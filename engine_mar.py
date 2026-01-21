import math,pdb
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import shutil
import cv2
import numpy as np
import os,time
import copy
import time
from torchvision.utils import save_image
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
F = torch.nn.functional

def gather_imgs(dataset, sidx, eidx):
    return torch.stack([dataset.images[i] for i in range(sidx, eidx)])

def plot(predict_img, test_img, output_dir='', epoch = 0, sidx = 0):
    # pred: B * T * H * W 
    # test_img: (1) * T * H * W 
    x_min = test_img.min()
    x_max = test_img.max()
    predict_img = predict_img.clamp(x_min, x_max)
    predict_img = (predict_img - x_min) / (x_max - x_min)
    test_img = (test_img - x_min) / (x_max - x_min)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    diff = (predict_img - test_img[None,:,:])
    predict_img = torch.stack([predict_img, diff], dim=1).flatten(1,2).reshape(predict_img.size(0)*2, *predict_img.size()[1:])
    output = torch.cat([test_img[None,:,:], predict_img], dim=0)
    output = torch.tensor(mpl.colormaps['bwr'](output.flatten(0,1).cpu().data.numpy())[:, :, :, :3]).permute(0, 3, 1, 2)
    save_image(output, f'{output_dir}/plots/{epoch:04d}_{sidx}.png', nrow = predict_img.shape[1], normalize = True)

@torch.no_grad()
def roll_out_seq(model, init_frame, rollout_step, min_val = None, max_val = None):
    frame_pred = model.module.sample(init_frame, ema = True)
    out_list = []
    for i in tqdm(range(int(rollout_step))):
        frame_pred = model.module.sample(frame_pred[0], frame_pred[1], frame_pred[2])
        out_list.append(frame_pred[0].cpu().data.numpy())
    all = np.concatenate(out_list)[:,0]
    if min_val is None:
        min_val = all.min()
    if max_val is None:
        max_val = all.max()
    return mpl.colormaps['bwr']((all - min_val) / (max_val - min_val))[...,:3]

def eval_model_rec(model, data_loader, epoch=5000, device=0, sidx=5, tms=20, btz=1, output_dir='output_dir', \
                prefix = '', test=False, save_folder = 'data_long'):
    num_of_frame = 1
    if tms < 2000:
        test_img = gather_imgs(data_loader, sidx, sidx+tms*num_of_frame+1).float().to(device)
    else:
        test_img = gather_imgs(data_loader, sidx, sidx+1000).float().to(device)
    x_min = test_img.min()
    x_max = test_img.max()

    def video_sampling(init_cond, times, x_min = None, x_max = None, x_start = -1, soda_feat = None):
        x_min, x_max = (init_cond.min(), init_cond.max()) if x_min is None and x_max is None else (x_min, x_max)
        sample_img = [(init_cond[:, None, :, :] -x_min)/(x_max-x_min)]

        pred = init_cond[:,None]
        pred_mid_res = None
        pred_low_res = None

        with torch.cuda.amp.autocast():
            for t in tqdm(range(times+1)):
                pred, pred_mid_res, pred_low_res = model.module.sample(pred, pred_mid_res, pred_low_res)
                if tms >= 10000:
                    if t % 100 == 0:
                        torch.save({'pred':pred, 'pred_mid_res':pred_mid_res, 'pred_low_res':pred_low_res}, f'{save_folder}/{t:07d}.pth')
                else:
                    if t > 0: ### initial corner case
                        sample_img.append((pred[:,[0]] -x_min)/(x_max-x_min))
        if tms < 10000:
            return torch.cat(sample_img, dim=1)

    def eval_mse(predict, test_img):
        ## output shape: tms x 1
        mse = ((predict - test_img[None,:,:])**2).reshape(predict.shape[0], predict.shape[1], -1).mean(dim=2)
        return mse.mean(dim=0)[1:], mse.std(dim=0)[1:]

    def plot(predict_img, extra_prefix = ''):
        # pred: B * T * H * W 
        # test_img: (1) * T * H * W 
        diff = (predict_img - test_img[None,:,:])
        predict_img = torch.stack([predict_img, diff], dim=1).flatten(1,2).reshape(predict_img.size(0)*2, *predict_img.size()[1:])
        output = torch.cat([test_img[None,:,:], predict_img], dim=0)
        output = torch.tensor(mpl.colormaps['bwr'](output.flatten(0,1).cpu().data.numpy())[:, :, :, :3]).permute(0, 3, 1, 2)
        #print(output_di, epoch, sidx, prefix, extra_prefix, output.shape)
        save_image(output, f'{output_dir}/plots/{epoch:04d}_{sidx}{prefix}_{extra_prefix}.png', nrow = tms*num_of_frame+1, normalize = True)

    predict_img = video_sampling(test_img[0].repeat(btz, 1, 1), tms, x_min=x_min, x_max=x_max)

    ## normlize and calculate the diff
    if tms < 2000:
        os.makedirs(f'{output_dir}/csv', exist_ok=True)
        test_img = (test_img-x_min) / (x_max-x_min)
        mse, mse_std = eval_mse(predict_img, test_img)
        df = pd.DataFrame(dict(time=list(np.arange(tms*num_of_frame)), mse=mse.cpu().data.numpy(), mse_std=mse_std.cpu().data.numpy()))
        df.to_csv(f'{output_dir}/csv/mse_{epoch:04d}_{sidx}_{prefix}.csv', index=False, float_format='%.5f')
        return {'test_img': test_img, 'predict_img': predict_img, 'mse': mse, 'min_max': [x_min, x_max]}

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def train_one_epoch(model, 
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, 
                    dataset_validation = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 12

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    #img_random_crop = img_crop(scale=(0.8, 1.0), ratio=(0.75, 1.25))
    model.train()
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        x = samples.to(args.device).float()

        with torch.cuda.amp.autocast():
            loss, high_loss, mid_loss, low_loss = model(x) 
                 
        loss_value = loss.item()
        
        metric_logger.update(loss = loss_value)
        metric_logger.update(high_loss = high_loss.item())
        metric_logger.update(mid_loss = mid_loss.item())
        metric_logger.update(low_loss = low_loss.item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        grad_norm = loss_scaler(3 * loss, optimizer, clip_grad=None, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        model.module.ema_step(args.ema_rate)

        metric_logger.update(grad_norm=grad_norm.mean())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        pass

    if  epoch % (args.eval_freq) == 0 and args.gpu == 0 and epoch > 0:
        model.eval()
        eval_model_rec(model, data_loader.dataset, epoch, x.device, 0, prefix = 'train', output_dir=args.output_dir) 
        eval_model_rec(model, data_loader.dataset, epoch, x.device, 100, prefix = 'train', output_dir=args.output_dir) 
        eval_model_rec(model, dataset_validation, epoch, x.device, 0, output_dir=args.output_dir) 
        eval_model_rec(model, dataset_validation, epoch, x.device, 100, output_dir=args.output_dir) 
        model.train()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, x[:,[0]]