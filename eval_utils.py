import torch,pdb
import numpy as np
import scipy.io
import torch.nn as nn
import operator,pdb
from functools import reduce
import matplotlib as mpl

# PCA
class PCA(object):
    def __init__(self, x, dim, subtract_mean=True):
        super(PCA, self).__init__()

        # Input size
        x_size = list(x.size())

        # Input data is a matrix
        assert len(x_size) == 2

        # Reducing dimension is less than the minimum of the
        # number of observations and the feature dimension
        assert dim <= min(x_size)

        self.reduced_dim = dim

        if subtract_mean:
            self.x_mean = torch.mean(x, dim=0).view(1, -1)
        else:
            self.x_mean = torch.zeros((x_size[1],), dtype=x.dtype, layout=x.layout, device=x.device)

        # SVD
        U, S, V = torch.svd(x - self.x_mean)
        V = V.t()

        # Flip sign to ensure deterministic output
        max_abs_cols = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_cols, range(U.size()[1])]).view(-1, 1)
        V *= signs

        self.W = V.t()[:, 0:self.reduced_dim]
        self.sing_vals = S.view(-1, )

    def cuda(self):
        self.W = self.W.cuda()
        self.x_mean = self.x_mean.cuda()
        self.sing_vals = self.sing_vals.cuda()

    def encode(self, x):
        return (x - self.x_mean).mm(self.W)

    def decode(self, x):
        return x.mm(self.W.t()) + self.x_mean

    def forward(self, x):
        return self.decode(self.encode(x))

    def __call__(self, x):
        return self.forward(x)

@torch.no_grad()
def compute_pca_correlation(gt_seq, seq_list, PCA_dim = 50):
    # gt_seq: T * H * W
    # seq_list: [T * H * W]

    T = gt_seq.shape[0]
    x_pca = PCA(gt_seq.flatten(1,-1), min(T, PCA_dim), subtract_mean=False)
    pca_truth = x_pca.encode(gt_seq.flatten(1,-1))[:, 0].cpu().data.numpy()
    gt_coorelate = np.correlate(pca_truth, pca_truth, mode='full')[T-1:]
    out_list = []
    for seq in seq_list:
        pca_seq = x_pca.encode(seq.flatten(1,-1))[:, 0].cpu().data.numpy()
        seq_coorelate = np.correlate(pca_seq, pca_seq, mode='full')[T-1:]
        out_list.append(seq_coorelate)
    return gt_coorelate, out_list

def generate_gif(image_seq, file_name):
    from PIL import Image, ImageDraw
    T = image_seq.shape[0]
    img_seq = mpl.colormaps['bwr'](image_seq.cpu().data.numpy())[:, :, :, :3]
    frames = [Image.fromarray((img_seq[i] * 255).astype(np.uint8)) for i in range(T)]
    frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=100, loop=0)
