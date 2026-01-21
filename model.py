import torch
import torch.nn as nn
import math,pdb
import einops
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import copy
from util.data_losses import H1Loss
from ufno import UNet

F = torch.nn.functional

def get_hook_fn():
    data_list = []
    def hook_fn(module, input, output):
        data_list.append(output.norm(dim = -1).mean().item())
    return hook_fn, data_list

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

class FutureINN(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self.num_frame = args.num_of_frame
        self.ds_factor = args.ds_factor
        self.ds_res = args.img_size // self.ds_factor
        self.ds_factor_low = args.ds_factor_low
        self.low_res = args.img_size // self.ds_factor_low
        self.vit = UNet(in_channels = 1, out_channels = 1, inject_res = [64, 32], output_res = [self.ds_res, self.low_res])
        self.ema_vit = UNet(in_channels = 1, out_channels = 1, inject_res = [64, 32], output_res = [self.ds_res, self.low_res])
        self.ratio = args.ratio

    def forward(self, img_seq):
        b = img_seq.shape[0]

        img_seq_ds = F.avg_pool2d(img_seq, kernel_size = self.ds_factor, stride = self.ds_factor)
        img_seq_ds_2 = F.avg_pool2d(img_seq, kernel_size = self.ds_factor_low, stride = self.ds_factor_low)

        ## high resolution, low resolution, indicator
        ## input_1: boundary case
        ## input_2: boundary case
        ## input_3: boundary case
        input_1 = (img_seq[:,[0]], torch.zeros_like(img_seq_ds[:,[0]]), torch.zeros_like(img_seq_ds_2[:, [0]]), torch.zeros(img_seq.shape[0]).to(img_seq.device))
        input_2 = (img_seq[:,[0]], img_seq_ds[:,[1]], torch.zeros_like(img_seq_ds_2[:, [0]]), torch.ones(img_seq.shape[0]).to(img_seq.device))
        input_3 = (img_seq[:,[0]], img_seq_ds[:,[1]], img_seq_ds_2[:, [2]], 2*torch.ones(img_seq.shape[0]).to(img_seq.device))

        ## target_1: boundary case 
        ## target_2: boundary case
        ## target_3: autoregressive training
        target_1 = (torch.zeros_like(img_seq[:,[1]]), img_seq_ds[:,[1]], torch.zeros_like(img_seq_ds_2[:, [2]]))
        target_2 = (torch.zeros_like(img_seq[:,[1]]), torch.zeros_like(img_seq_ds[:,[1]]), img_seq_ds_2[:, [2]])
        target_3 = (img_seq[:,[1]], img_seq_ds[:,[2]], img_seq_ds_2[:, [3]])

        ## sampling the probability of the boundary case to predict z^{(1)}, z^{(2)} given the initial condition u.
        mask = F.one_hot(torch.multinomial(torch.FloatTensor([(1-self.ratio)/2, (1-self.ratio)/2, self.ratio]), b, replacement = True), num_classes = 3)[:,:,None,None,None].to(img_seq.device)

        input = (input_1[0] * mask[:,0] + input_2[0] * mask[:,1] + input_3[0]*mask[:,2],
                 input_1[1] * mask[:,0] + input_2[1] * mask[:,1] + input_3[1]*mask[:,2],
                 input_1[2] * mask[:,0] + input_2[2] * mask[:,1] + input_3[2]*mask[:,2],
                 input_1[3] * mask[:,0].reshape(-1) + input_2[3] * mask[:,1].reshape(-1) + input_3[3]*mask[:,2].reshape(-1))

        target = (target_1[0] * mask[:,0] + target_2[0]*mask[:,1] + target_3[0]*mask[:,2],
                  target_1[1] * mask[:,0] + target_2[1]*mask[:,1] + target_3[1]*mask[:,2],
                  target_1[2] * mask[:,0] + target_2[2]*mask[:,1] + target_3[2]*mask[:,2])

        out_high_res, out_mid_res, out_low_res = self.vit(*input)
        high_loss = ((target[0] - out_high_res) ** 2) + (target[0] - out_high_res).abs()
        mid_loss = ((target[1] - out_mid_res) ** 2) + (target[1] - out_mid_res).abs()
        low_loss = ((target[2] - out_low_res) ** 2) + (target[2] - out_low_res).abs()
        high_loss = high_loss * (mask[:,2] == 1).float()
        mid_loss = mid_loss * (mask[:,1] != 1).float()
        low_loss = low_loss * (mask[:,0] != 1).float()
        loss = (2*high_loss).mean() + mid_loss.mean() + low_loss.mean()

        return loss, (2*high_loss).mean(), mid_loss.mean(), low_loss.mean()

    def ema_step(self, ema_ratio = 0.9995):
        update_ema(self.ema_vit, self.vit, ema_ratio)

    @torch.no_grad()
    def sample(self, high_x, mid_x = None, low_x = None, ema=True):
        if ema:
            model = self.ema_vit
        else:
            model = self.vit
        b = high_x.shape[0]
        self.eval()
        with torch.no_grad():
            if mid_x is None:
                with torch.cuda.amp.autocast():
                    _, mid_x, _ = model(high_x, torch.zeros(b, 1,  self.ds_res,  self.ds_res).float().to(high_x), \
                                                torch.zeros(b, 1,  self.low_res,  self.low_res).float().to(high_x), torch.zeros(b).to(high_x)) 
                    _, _, low_x = model(high_x, mid_x, \
                                                torch.zeros(b, 1,  self.low_res,  self.low_res).float().to(high_x), torch.ones(b).to(high_x)) 
                    ## we don't update high_x
                    high_x = high_x
            else:
                with torch.cuda.amp.autocast():
                    high_x, mid_x, low_x = model(high_x, mid_x, low_x, 2*torch.ones(b).to(high_x)) 
        self.train()
        return high_x, mid_x, low_x
