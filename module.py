import torch
import torch.nn as nn

def calc_padding(kernel_size, dilation, causal, stride=1):
    if causal:
        padding = (kernel_size-1)*dilation+1-stride
    else:
        padding = ((kernel_size-1)*dilation+1-stride)//2
    return padding

class ConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd, dl=1):
        super(ConvGLU1D, self).__init__()
        self.padding = calc_padding(ks,dl,False,sd)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, stride=sd, padding=self.padding, dilation=dl)
        self.bn1 = nn.BatchNorm1d(out_ch*2)
    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

class DeconvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd, dl=1):
        super(DeconvGLU1D, self).__init__()
        self.padding = calc_padding(ks,dl,False,sd)
        self.conv1 = nn.ConvTranspose1d(
            in_ch, out_ch*2, ks, stride=sd, padding=self.padding, dilation=dl)
        self.bn1 = nn.BatchNorm1d(out_ch*2)
    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

class LinearWN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearWN, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.linear_layer = nn.utils.weight_norm(self.linear_layer)

    def forward(self, x):
        return self.linear_layer(x)

def concat_dim1(x,y):
    aux_ch = len(y)
    if torch.Tensor.dim(x) == 3:
        y0 = torch.unsqueeze(torch.unsqueeze(y,0),2)
        N, n_ch, n_t = x.shape
        yy = y0.repeat(N,1,n_t)
        h = torch.cat((x,yy), dim=1)
    elif torch.Tensor.dim(x) == 4:
        y0 = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(y,0),2),3)
        N, n_ch, n_q, n_t = x.shape
        yy = y0.repeat(N,1,n_q,n_t)
        h = torch.cat((x,yy), dim=1)
    
    return h

def concat_dim2(x,y):
    aux_ch = len(y)
    if torch.Tensor.dim(x) == 3:
        y0 = torch.unsqueeze(torch.unsqueeze(y,0),0)
        N, n_t, n_ch = x.shape
        yy = y0.repeat(N,n_t,1)
        h = torch.cat((x,yy), dim=2)
    elif torch.Tensor.dim(x) == 2:
        y0 = torch.unsqueeze(y,0)
        N, n_ch = x.shape
        yy = y0.repeat(N,1)
        h = torch.cat((x,yy), dim=1)
    
    return h