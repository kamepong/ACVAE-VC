# Copyright 2021 Hirokazu Kameoka

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import module as md
   
class Encoder1(nn.Module):
    # 1D Strided Convolution
    def __init__(self, in_ch, clsnum, out_ch, mid_ch):
        super(Encoder1, self).__init__()
        self.le1 = md.ConvGLU1D(in_ch+clsnum, mid_ch, 9, 1)
        self.le2 = md.ConvGLU1D(mid_ch+clsnum, mid_ch, 8, 2)
        self.le3 = md.ConvGLU1D(mid_ch+clsnum, mid_ch, 8, 2)
        self.le4 = nn.Conv1d(mid_ch+clsnum, out_ch*2, 5, stride=1, padding=(5-1)//2)

    def __call__(self, xin, y):
        device = xin.device
        N, n_ch_in, n_t_in = xin.shape
        n_t_new = math.ceil(n_t_in/4)*4
        if n_t_new!=n_t_in:
            padnum = n_t_new - n_t_in
            zeropad = np.zeros((N,n_ch_in,padnum))
            zeropad = torch.tensor(zeropad).to(device, dtype=torch.float)
            xin = torch.cat((xin,zeropad),dim=2)

        out = xin
        out = md.concat_dim1(out,y)
        out = self.le1(out)
        out = md.concat_dim1(out,y)
        out = self.le2(out)
        out = md.concat_dim1(out,y)
        out = self.le3(out)
        out = md.concat_dim1(out,y)
        out = self.le4(out)

        out = out.clone()[:,:,0:n_t_in]
        mu, ln_var = torch.split(out,out.shape[1]//2,dim=1)

        ln_var = torch.clamp(ln_var, min = -50.0, max = 0.0)
        
        return mu, ln_var

class Encoder2(nn.Module):
    # MLP+BiLSTM+MLP
    def __init__(self, in_ch, clsnum, out_ch, mid_ch, num_layers=2, negative_slope=0.1):
        super(Encoder2, self).__init__()

        self.start = md.LinearWN(in_ch+clsnum, mid_ch)
        self.lrelu0 = nn.LeakyReLU(negative_slope)
        self.rnn = nn.LSTM(
            mid_ch+clsnum,
            mid_ch//2,
            num_layers,
            dropout=0,
            bidirectional=True,
            batch_first = True
        )
        self.end = md.LinearWN(mid_ch+clsnum, out_ch*2)

    def __call__(self, xin, y):
        device = xin.device
        num_batch, num_mels, num_frame = xin.shape

        out = xin.permute(0,2,1) # (num_batch, num_frame, num_mels)
        out = md.concat_dim2(out,y) # (num_batch, num_frame, num_mels+clsnum)
        out = self.lrelu0(self.start(out))
        out = md.concat_dim2(out,y)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        out = md.concat_dim2(out,y)
        out = self.end(out)
        out = out.permute(0,2,1) # (num_batch, out_ch*2, num_frame)

        mu, ln_var = torch.split(out,out.shape[1]//2,dim=1)

        ln_var = torch.clamp(ln_var, min = -50.0, max = 0.0)
        
        return mu, ln_var
    
class Decoder1(nn.Module):
    # 1D Strided Convolution
    def __init__(self, in_ch, clsnum, out_ch, mid_ch):
        super(Decoder1, self).__init__()
        self.le1 = md.DeconvGLU1D(in_ch+clsnum, mid_ch, 5, 1)
        self.le2 = md.DeconvGLU1D(mid_ch+clsnum, mid_ch, 8, 2)
        self.le3 = md.DeconvGLU1D(mid_ch+clsnum, mid_ch, 8, 2)
        pad4 = md.calc_padding(9,1,False,stride=1)
        self.le4 = nn.ConvTranspose1d(mid_ch+clsnum, out_ch*2, 9, stride=1, padding=pad4)

    def __call__(self, zin, y, num_frames=None):
        device = zin.device

        out = zin
        out = md.concat_dim1(out,y)
        out = self.le1(out)
        out = md.concat_dim1(out,y)
        out = self.le2(out)
        out = md.concat_dim1(out,y)
        out = self.le3(out)
        out = md.concat_dim1(out,y)
        out = self.le4(out)
        
        if num_frames is not None:
            out = out.clone()[:,:,0:num_frames]
        mu, ln_var = torch.split(out,out.shape[1]//2,dim=1)

        ln_var = torch.clamp(ln_var, min = -50.0, max = 0.0)

        return mu, ln_var

class Decoder2(nn.Module):
    # MLP+BiLSTM+MLP
    def __init__(self, in_ch, clsnum, out_ch, mid_ch, num_layers=2, negative_slope=0.1):
        super(Decoder2, self).__init__()
        
        self.start = md.LinearWN(in_ch+clsnum, mid_ch)
        self.lrelu0 = nn.LeakyReLU(negative_slope)
        self.rnn = nn.LSTM(
            mid_ch+clsnum,
            mid_ch//2,
            num_layers,
            dropout=0,
            bidirectional=True,
            batch_first = True
        )
        self.end = md.LinearWN(mid_ch+clsnum, out_ch*2)

    def __call__(self, zin, y, num_frames=None):
        device = zin.device
        num_batch, z_dim, num_frame = zin.shape

        out = zin.permute(0,2,1) # (num_batch, num_frame, z_dim)
        out = md.concat_dim2(out,y) # (num_batch, num_frame, z_dim+clsnum)
        out = self.lrelu0(self.start(out))
        out = md.concat_dim2(out,y)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(out)
        out = md.concat_dim2(out,y)
        out = self.end(out)
        out = out.permute(0,2,1) # (num_batch, z_dim*2, num_frame)

        mu, ln_var = torch.split(out,out.shape[1]//2,dim=1)

        ln_var = torch.clamp(ln_var, min = -50.0, max = 0.0)
        
        return mu, ln_var

class Classifier1(nn.Module):
    # 1D Strided Convolution
    def __init__(self, in_ch, clsnum, mid_ch, dor=0.2):
        super(Classifier1, self).__init__()
        self.le1 = md.ConvGLU1D(in_ch,mid_ch,5,1)
        self.le2 = md.ConvGLU1D(mid_ch,mid_ch,4,2)
        self.le3 = md.ConvGLU1D(mid_ch,mid_ch,4,2)
        self.le4 = md.ConvGLU1D(mid_ch,mid_ch,4,2)
        self.le5 = nn.Conv1d(mid_ch, clsnum, 5, stride=1, padding=(5-1)//2)
        self.do1 = nn.Dropout(p=dor)
        self.do2 = nn.Dropout(p=dor)
        self.do3 = nn.Dropout(p=dor)
        self.do4 = nn.Dropout(p=dor)
        self.sm1 = nn.Softmax(dim=1)

    def __call__(self, xin):
        device = xin.device
        N, n_ch_in, n_t_in = xin.shape
        n_t_new = math.ceil(n_t_in/8)*8
        if n_t_new!=n_t_in:
            padnum = n_t_new - n_t_in
            zeropad = np.zeros((N,n_ch_in,padnum))
            zeropad = torch.tensor(zeropad).to(device, dtype=torch.float)
            xin = torch.cat((xin,zeropad),dim=2)

        out = xin
        out = self.do1(self.le1(out))
        out = self.do2(self.le2(out))
        out = self.do3(self.le3(out))
        out = self.do4(self.le4(out))
        d = self.le5(out)

        p = self.sm1(d)

        return d, p
    
class ACVAE(nn.Module):
    def __init__(self, enc, dec, cls):
        super(ACVAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.cls = cls

    def gaussian(self, z_mu, z_lnvar):
        device = z_mu.device
        epsilon = torch.randn(z_mu.shape).to(device, dtype=torch.float)
        z = z_mu + torch.sqrt(torch.exp(z_lnvar)) * epsilon
        return z
        
    def gaussian_kl_divergence(self, z_mu, z_lnvar):
        kl = -0.5 * torch.mean(1 + z_lnvar - z_mu**2 - torch.exp(z_lnvar))
        return kl

    def gaussian_nll(self, x, x_mu, x_lnvar):
        x_prec = torch.exp(-x_lnvar)
        x_diff = x - x_mu
        x_power = (x_diff * x_diff) * x_prec * -0.5
        like_loss = torch.mean((x_lnvar + math.log(2 * math.pi)) / 2 - x_power)
        return like_loss

    def __call__(self, x_s, l_s, l_t):
        N_s, n_ch_s, n_frame_s =  x_s.shape
        return self.dec(self.enc(x_s, l_s)[0], l_t, n_frame_s)[0]

    def calc_loss(self, x_s, x_t, clsind_s, clsind_t, clsnum):
        device = x_s.device
        N_s, n_ch_s, n_frame_s =  x_s.shape
        N_t, n_ch_t, n_frame_t =  x_t.shape
        n_y = clsnum
        L_s = np.eye(n_y,dtype=np.int)[clsind_s]
        L_t = np.eye(n_y,dtype=np.int)[clsind_t]
        
        l_s = torch.tensor(L_s).to(device, dtype=torch.float)
        l_t = torch.tensor(L_t).to(device, dtype=torch.float)
        
        # Encode x_s
        z_mu_s, z_ln_var_s = self.enc(x_s, l_s)
        z_s = self.gaussian(z_mu_s, z_ln_var_s)
        #s2s reconstruction
        x_mu_ss, x_ln_var_ss = self.dec(z_s, l_s, n_frame_s)
        xf_ss = self.gaussian(x_mu_ss, x_ln_var_ss)
        #s2t conversion
        x_mu_st, x_ln_var_st = self.dec(z_s, l_t, n_frame_s)
        xf_st = self.gaussian(x_mu_st, x_ln_var_st)

        # Encode x_t
        z_mu_t, z_ln_var_t = self.enc(x_t, l_t)
        z_t = self.gaussian(z_mu_t, z_ln_var_t)
        #t2s conversion
        x_mu_ts, x_ln_var_ts = self.dec(z_t, l_s, n_frame_t)
        xf_ts = self.gaussian(x_mu_ts, x_ln_var_ts)
        #t2t reconstruction
        x_mu_tt, x_ln_var_tt = self.dec(z_t, l_t, n_frame_t)
        xf_tt = self.gaussian(x_mu_tt, x_ln_var_tt)
        
        # VAE loss
        vae_loss_prior_s = self.gaussian_kl_divergence(z_mu_s, z_ln_var_s)
        vae_loss_like_s = self.gaussian_nll(x_s, x_mu_ss, x_ln_var_ss)

        vae_loss_prior_t = self.gaussian_kl_divergence(z_mu_t, z_ln_var_t)
        vae_loss_like_t = self.gaussian_nll(x_t, x_mu_tt, x_ln_var_tt)

        vae_loss_prior = (vae_loss_prior_s + vae_loss_prior_t)/2.0
        vae_loss_like = (vae_loss_like_s + vae_loss_like_t)/2.0
        
        # Domain classification loss defined as cross entropy
        # The smaller this value becomes, the greater the likelihood.
        dcr_s, _ = self.cls(x_s)
        dcr_t, _ = self.cls(x_t)
        dcf_ss, _ = self.cls(xf_ss)
        dcf_st, _ = self.cls(xf_st)
        dcf_tt, _ = self.cls(xf_tt)
        dcf_ts, _ = self.cls(xf_ts)
        # dcr_s: N x n_y x T array

        dcr_s = dcr_s.permute(0,2,1)
        dcr_t = dcr_t.permute(0,2,1)
        dcf_ss = dcf_ss.permute(0,2,1)
        dcf_st = dcf_st.permute(0,2,1)
        dcf_tt = dcf_tt.permute(0,2,1)
        dcf_ts = dcf_ts.permute(0,2,1)
        # dcr_s: N x T x n_y array
        
        dcr_s = torch.reshape(dcr_s, (-1,n_y))
        dcr_t = torch.reshape(dcr_t, (-1,n_y))
        dcf_ss = torch.reshape(dcf_ss, (-1,n_y))
        dcf_st = torch.reshape(dcf_st, (-1,n_y))
        dcf_tt = torch.reshape(dcf_tt, (-1,n_y))
        dcf_ts = torch.reshape(dcf_ts, (-1,n_y))
        # dcr_s: NT x n_y array

        # cr_s is a class label sequence with the length of NT
        cr_s = torch.tensor(clsind_s*np.ones(len(dcr_s))).to(device, dtype=torch.long)
        cr_t = torch.tensor(clsind_t*np.ones(len(dcr_t))).to(device, dtype=torch.long)
        cf_ss = torch.tensor(clsind_s*np.ones(len(dcf_ss))).to(device, dtype=torch.long)
        cf_st = torch.tensor(clsind_t*np.ones(len(dcf_st))).to(device, dtype=torch.long)
        cf_tt = torch.tensor(clsind_t*np.ones(len(dcf_tt))).to(device, dtype=torch.long)
        cf_ts = torch.tensor(clsind_s*np.ones(len(dcf_ts))).to(device, dtype=torch.long)
        # cr_s: NT-dimensional vector
        
        ClsLoss_r = (F.cross_entropy(dcr_s, cr_s)*dcr_s.shape[0]+
                     F.cross_entropy(dcr_t, cr_t)*dcr_t.shape[0])/(dcr_s.shape[0]+
                                                                   dcr_t.shape[0])

        #ClsLoss_f = 0
        ClsLoss_f = (F.cross_entropy(dcf_ss, cf_ss)*dcf_ss.shape[0] +
                     F.cross_entropy(dcf_st, cf_st)*dcf_st.shape[0] +
                     F.cross_entropy(dcf_tt, cf_tt)*dcf_tt.shape[0] +
                     F.cross_entropy(dcf_ts, cf_ts)*dcf_ts.shape[0])/(dcf_ss.shape[0]+
                                                                      dcf_st.shape[0]+
                                                                      dcf_tt.shape[0]+
                                                                      dcf_ts.shape[0])

        return vae_loss_prior, vae_loss_like, ClsLoss_r, ClsLoss_f

