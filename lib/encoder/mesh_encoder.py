import torch
import torch.nn as nn


class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True, device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class SpiralEncoder(nn.Module):
    def __init__(self, filters_enc, latent_size, sizes, spiral_sizes,
                 spirals, D, U, activation='elu', device=None):
        super(SpiralEncoder, self).__init__()
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.spiral_sizes = spiral_sizes
        self.sizes = sizes
        self.D = D
        self.U = U
        self.activation = activation
        self.device = device

        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes) - 1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i + 1],
                                        activation=self.activation, device=device))
            input_size = filters_enc[0][i + 1]

        self.conv = nn.ModuleList(self.conv)

        self.fc_latent_enc = nn.Linear((self.sizes[-1] + 1) * input_size, latent_size)

    def encode(self, x):
        if len(x.shape) == 4:
            batch_size, n_steps, n_pts, _ = x.shape
            x = x.transpose(1, 2).contiguous().view(batch_size, n_pts, -1)
        bsize = x.size(0)
        S = [torch.from_numpy(s).long().to(self.device) for s in self.spirals]
        D = [torch.from_numpy(s).float().to(self.device) for s in self.D]
        # S = self.spirals
        # D = self.D

        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
        x = x.view(bsize, -1)
        return self.fc_latent_enc(x)

    def forward(self, x):
        x = self.encode(x)
        return x

