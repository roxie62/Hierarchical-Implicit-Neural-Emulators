import numpy as np
import torch
import scipy.io
import pdb,tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist

from py2d import NSdataset

F = torch.nn.functional


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def parallel_load_image(dataset, img_res, ngpu, data_norm):
    def collate_fn(batch):
        return [F.interpolate(torch.from_numpy(data)[None,None], mode = 'bilinear', size = (img_res, img_res), antialias = True)[0,0] / data_norm for data in batch ]
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None,
        batch_size=256,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    images_list = []
    for images in tqdm.tqdm(data_loader):
        images_list += images
        if ngpu == 1:
            if len(images_list) > 4000:
                break
    return images_list

class DistributedPreloadDataset(torch.utils.data.Dataset):
    def __init__(self, frame_num = 5, img_res = 512, expand = 1, eval=False, validation=False):
        self.expand = expand
        self.dataset = NSdataset(get_UV=False, get_Psi=False, normalize=False, eval=eval, validation=validation)
        self.frame_num = frame_num
        world_size = get_world_size()
        rank = get_rank()
        idx = np.arange(len(self.dataset))
        chunk_size = (idx.shape[0] + world_size - 1) // world_size
        image_chunk = idx[chunk_size * rank: chunk_size * (rank+1)]
        self.dataset.extra_index = image_chunk
        self.data_norm = 10
        self.images = parallel_load_image(self.dataset, img_res, world_size, self.data_norm) 
        self.max_frame = 50
        torch.distributed.barrier()
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.index = torch.randperm(len(self.images) - self.frame_num + 1)

    def __len__(self):
        return len(self.index)  * self.expand
        
    def __getitem__(self, index):
        index = self.index[index // self.expand]
        return torch.stack([self.images[index + i] for i in range(self.frame_num)], dim = 0)

if __name__ == '__main__':
    dataset = NSdataset(get_UV=True, get_Psi=True, normalize=False)