import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
)

class DecoderCBatchNorm(nn.Module):
    ''' Decoder class with CBN for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_size (int): hidden dimension
        leaky (bool): whether to use leaky activation
        legacy (bool): whether to use legacy version
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.dim = dim

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    # For ONet 4D
    def add_time_axis(self, p, t):
        ''' Adds time axis to points.

        Args:
            p (tensor): points
            t (tensor): time values
        '''
        n_pts = p.shape[1]
        t = t.unsqueeze(1).repeat(1, n_pts, 1)
        p_out = torch.cat([p, t], dim=-1)
        return p_out

    def forward(self, p, c, **kwargs):
        ''' Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned temporal code c
        '''
        # if p.shape[-1] != self.dim:
        #     p = self.add_time_axis(p, kwargs['t'])

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
