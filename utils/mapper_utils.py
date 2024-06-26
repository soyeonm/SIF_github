import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    mt = - t 
    cos_mt = mt.cos()
    sin_mt = mt.sin()

    mtheta11 = torch.stack([cos_mt, -sin_mt,
                           torch.zeros(cos_mt.shape).float().to(device)], 1)
    mtheta12 = torch.stack([sin_mt, cos_mt,
                           torch.zeros(cos_mt.shape).float().to(device)], 1)
    mtheta1 = torch.stack([mtheta11, mtheta12], 1)
    #breakpoint()
    mtheta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), -x], 1)
    mtheta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), -y], 1)
    mtheta2 = torch.stack([mtheta21, mtheta22], 1)

    inverse_rot_grid = F.affine_grid(mtheta1, torch.Size(grid_size))
    inverse_trans_grid = F.affine_grid(mtheta2, torch.Size(grid_size))

    return rot_grid, trans_grid, inverse_rot_grid, inverse_trans_grid


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)



# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
