import sys
import math
import time
import queue
import random
import h5py as hp
import numpy as np
import scipy.misc
import scipy.io as sio
import scipy.ndimage
import skimage.feature
import matplotlib.pyplot as plt

import pywt
import os
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from threading import Thread  # needed since the denoiser is running in parallel
from kymatio.torch import Scattering2D
from transformations import rotation_matrix, translation_matrix
from skimage.restoration import denoise_nl_means, estimate_sigma
from pywt import waverecn, wavedecn

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
dtype = torch.cuda.FloatTensor



### REGULARIZATION
def tv_2d(img, tv_weight):
    h_variance = torch.sum(torch.pow(img[:, :, :, :-1, :] - img[:, :, :, 1:, :], 2))
    w_variance = torch.sum(torch.pow(img[:, :, :-1, :, :] - img[:, :, 1:, :, :], 2))
    
    loss = tv_weight * (h_variance + w_variance)
    
    return loss


def tv_3d(img, tv_weight):
    d_variance = torch.sum(torch.pow(img[:, :, :, :, :-1] - img[:, :, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :, :-1, :] - img[:, :, :, 1:, :], 2))
    w_variance = torch.sum(torch.pow(img[:, :, :-1, :, :] - img[:, :, 1:, :, :], 2))
    
    loss = tv_weight * (h_variance + w_variance + d_variance)
    
    return loss


def tukeywin(L, D, device_num, r = 0.03):
    win = torch.ones(L, )

    win[0 : int(r/2 * L)] = 0.5 * (1 + np.cos(2 * np.pi / r * (torch.arange(0, int(r/2 * L))/(L-1) - r/2)))
    win[L - int(r/2 * L) : L] = 0.5 * (1 + np.cos(2 * np.pi / r * (torch.arange(L - int(r/2 * L), L)/(L-1) - 1 + r/2)))

    win = win.unsqueeze(1) * win.unsqueeze(0)
    
    return win.unsqueeze(-1).repeat(1, 1, D).cuda(device_num)


def border_effect_reg(img, weight, device_num):
    L = img.shape[-2]
    D = img.shape[-1]
    
    img /= (tukeywin(L, D, device_num).unsqueeze(0).unsqueeze(0) + 1e-6)
    
    return weight * torch.mean(torch.pow(torch.abs(img), 2))


def calculate_energy(coeffs, level=3):
    energy = np.zeros((1+level*3,))
    
    energy[0] = np.mean(np.abs(coeffs[0]) ** 2)
    for indx in range(1, level+1):
        energy[3*(indx-1)+1] = np.mean(np.abs(coeffs[indx]['ad']) ** 2)
        energy[3*(indx-1)+2] = np.mean(np.abs(coeffs[indx]['da']) ** 2)
        energy[3*(indx-1)+3] = np.mean(np.abs(coeffs[indx]['dd']) ** 2)

    sum_energy = np.sum(energy)
    
    return sum_energy


def normalize_rec(rec, obj):
    if np.sum(np.abs(rec) ** 2) > np.finfo(float).eps:
        rec *= np.sqrt(np.sum(np.abs(obj) ** 2) / np.sum(np.abs(rec) ** 2))
    
    return rec


def wavelet_sino_hp(sino):
    rec = np.zeros_like(sino)
    
    # obj: Ny x Nx x Nz
    for idx in range(0, np.shape(sino)[-1]):
        level = 4
        
        f_aa = 1
        f_ad = [2.5, 2.5**2, 2.5**3, 2.5**4]
        f_da = [2.5, 2.5**2, 2.5**3, 2.5**4]
        f_dd = [2.5, 2.5**2, 2.5**3, 2.5**4]
            
        coeffs = wavedecn(sino[:, :, idx], 'db8', level = level)
        sum_energy_pre = calculate_energy(coeffs, level = level)

        coeffs[0] *= f_aa
        for indx in range(1, level+1):
            coeffs[indx]['ad'] *= f_ad[indx-1]
            coeffs[indx]['da'] *= f_da[indx-1]
            coeffs[indx]['dd'] *= f_dd[indx-1]

        sum_energy_post = calculate_energy(coeffs, level = level)  
        rec[:, :, idx] = normalize_rec(waverecn(coeffs, 'db8'), sino[:, :, idx])
        
    return rec


def wavelet_sino_lp(sino):
    rec = np.zeros_like(sino)
    
    # obj: Ny x Nx x Nz
    for idx in range(0, np.shape(sino)[-1]):
        level = 2
        f_aa = 4
        f_ad = [2, 1]
        f_da = [2, 1]
        f_dd = [2, 1]
            
        coeffs = wavedecn(sino[:, :, idx], 'db8', level = level)
        sum_energy_pre = calculate_energy(coeffs, level = level)

        coeffs[0] *= f_aa
        for indx in range(1, level+1):
            coeffs[indx]['ad'] *= f_ad[indx-1]
            coeffs[indx]['da'] *= f_da[indx-1]
            coeffs[indx]['dd'] *= f_dd[indx-1]

        sum_energy_post = calculate_energy(coeffs, level = level)  
#         coeffs = normalize_energy(coeffs, np.sqrt(sum_energy_pre / sum_energy_post), level = level)
        rec[:, :, idx] = normalize_rec(waverecn(coeffs, 'db8'), sino[:, :, idx])
        
    return rec


def pearsonr(inp, tar):
    # inp: M x N x D
    nom_p = inp - np.mean(inp, axis = (0, 1, 2))
    nom_q = tar - np.mean(tar, axis = (0, 1, 2))
    nom = np.mean(nom_p * nom_q, axis = (0, 1, 2))
    
    den_p = np.std(inp, axis = (0, 1, 2))
    den_q = np.std(tar, axis = (0, 1, 2))
    den = den_p * den_q
    
    return nom / den    



### UTIL FUNCTIONS
def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().detach().numpy()


def upsample(x):
    return F.interpolate(x.permute(2, 0, 1).unsqueeze(0), scale_factor = 2.0, 
                         mode = 'bicubic', align_corners = False).squeeze(0).permute(1, 2, 0)


def plot_and_save_figure(num_iter, rec, obj_H, loss_pred):
    plt.figure(figsize = (32, 20))
    
    plt.subplot(5,8,6); plt.plot(loss_pred, label = 'Training loss'); plt.legend()
    plt.subplot(5,8,9); plt.imshow(rec[0, 0, :, :, 0]); plt.colorbar()
    plt.subplot(5,8,11); plt.imshow(obj_H[0, 0, :, :, 0]); plt.colorbar()
    plt.subplot(5,8,13); plt.imshow(rec[0, 0, :, :, 5]); plt.colorbar()
    plt.subplot(5,8,15); plt.imshow(obj_H[0, 0, :, :, 5]); plt.colorbar()
    plt.subplot(5,8,17); plt.imshow(rec[0, 0, :, :, 21]); plt.colorbar()
    plt.subplot(5,8,19); plt.imshow(obj_H[0, 0, :, :, 21]); plt.colorbar()
    plt.subplot(5,8,21); plt.imshow(rec[0, 0, :, :, 72]); plt.colorbar()
    plt.subplot(5,8,23); plt.imshow(obj_H[0, 0, :, :, 72]); plt.colorbar()
    plt.subplot(5,8,25); plt.imshow(rec[0, 0, :, :, 108]); plt.colorbar()
    plt.subplot(5,8,27); plt.imshow(obj_H[0, 0, :, :, 108]); plt.colorbar()
    plt.subplot(5,8,29); plt.imshow(rec[0, 0, :, :, 113]); plt.colorbar()
    plt.subplot(5,8,31); plt.imshow(obj_H[0, 0, :, :, 113]); plt.colorbar()
    plt.subplot(5,8,33); plt.imshow(rec[0, 0, :, :, 131]); plt.colorbar()
    plt.subplot(5,8,35); plt.imshow(obj_H[0, 0, :, :, 131]); plt.colorbar()
    plt.subplot(5,8,37); plt.imshow(rec[0, 0, :, :, 136]); plt.colorbar()
    plt.subplot(5,8,39); plt.imshow(obj_H[0, 0, :, :, 136]); plt.colorbar()
    
    plt.savefig('../figures_6/DIP_PyXL_12092022_H/' + str(num_iter) + '_250_angles_5.png', 
                bbox_inches = 'tight')

    

### LOSS FUNCTIONS
def adaptive_tv_l1_loss_v1(img, w_L = 2e-6, w_M = 2e-7, w_R = 2e-8, 
                           separate_at_index_L = 100,
                           separate_at_index_R = 121):
    x = torch.arange(img.size(dim = -1))
    tv_weight = torch.cat((w_L * torch.ones((separate_at_index_L, )),
                           w_M * torch.ones((separate_at_index_R - separate_at_index_L, )),
                           w_R * torch.ones((img.size(dim = -1) - separate_at_index_R, )))).cuda(0)
    
    h_variance = torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]), dim = (0, 1))
    w_variance = torch.sum(torch.abs(img[:-1, :, :] - img[1:, :, :]), dim = (0, 1))
    
    loss = torch.sum(tv_weight * (h_variance + w_variance))
    
    return loss


def npcc_loss(y_pred, y_true, weight):
    up = torch.mean((y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true)))
    down = torch.std(y_pred) * torch.std(y_true)
    loss = 1 - up / down

    return weight * loss.type(dtype)



### FORWARD OPERATOR FOR PHYSICAL SYSTEM
def get_rot_mat(rot_mat, device_num):
    return torch.cuda.FloatTensor(rot_mat[0:-1, :]).unsqueeze(0).cuda(device_num)

def rot_vol(x, rot_mat, device_num):
    grid = F.affine_grid(rot_mat, x.size(), align_corners = False).cuda(device_num)
    x = F.grid_sample(x, grid, align_corners = False).cuda(device_num)
    return x

def forward_operator(obj, RANGE, device_num):
    pred_sino = []
    theta_lamino = 61.108 # deg
    mat_lamino = torch.cuda.FloatTensor(rotation_matrix(np.radians(theta_lamino), (0, 1, 0))).cuda(device_num)
    
    for rot_indx in RANGE:
#         print(rot_indx)
        theta_rot = 0.495 + np.rad2deg(angles[rot_indx] - angles[0]) # deg
        mat_theta = torch.cuda.FloatTensor(rotation_matrix(np.radians(90-theta_rot), (1, 0, 0))).cuda(device_num)
        
        rot_mat = get_rot_mat(torch.mm(mat_theta, mat_lamino), device_num)
        rot_obj = rot_vol(obj, rot_mat, device_num)
        
        pred_sino.append(torch.fliplr(torch.sum(rot_obj, -1).squeeze(0).squeeze(0)))
    
    return torch.stack(pred_sino, dim = -1)



### MODEL DEFINITION
class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        self.bn = nn.BatchNorm3d(out_channels, affine = True)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = (2, 2, 2)):
        super(DownBlock, self).__init__()
        
        if stride == (2, 2, 2):
            padding = 1
        else:
            padding = 'same'
            
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, 
                               padding = padding, bias = True)
        self.bn1 = nn.BatchNorm3d(out_channels, affine = True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = True)
        self.bn2 = nn.BatchNorm3d(out_channels, affine = True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x
    
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, high_pass = True, scale_factor = (2, 2, 2), learnable_weights = True):
        super(UpBlock, self).__init__()
        
        self.learnable_weights = learnable_weights
        self.high_pass = high_pass
        
        if self.learnable_weights:
            self.up = nn.Upsample(scale_factor = scale_factor, mode = 'nearest', align_corners = None)
            self.bn0 = nn.BatchNorm3d(in_channels, affine = True)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = True)
            self.bn1 = nn.BatchNorm3d(out_channels, affine = True)
            self.act = nn.ReLU()
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
            self.bn2 = nn.BatchNorm3d(out_channels, affine = True)
        
        else:
            self.up = nn.Upsample(scale_factor = scale_factor, mode = 'nearest', align_corners = None)
            self.bn0 = nn.BatchNorm3d(in_channels, affine = True)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = True)
            self.bn1 = nn.BatchNorm3d(out_channels, affine = True)
            self.act = nn.ReLU()
                
            if self.high_pass:
                self.conv_fh = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = False)
                self.bn_fh = nn.BatchNorm3d(out_channels, affine = False)
                self.conv_fv = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = False)
                self.bn_fv = nn.BatchNorm3d(out_channels, affine = False)
                
            else:
                self.conv_f = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = False)
                self.bn_f = nn.BatchNorm3d(out_channels, affine = False)
            
    def forward(self, x, skip): 
        if self.learnable_weights:
            x = self.up(x)     
            if skip is not None:
                x = torch.cat((x, skip), dim = 1)
            x = self.bn0(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            
            return x
        
        else:
            x = self.up(x)
            if skip is not None:
                x = torch.cat((x, skip), dim = 1)
            x = self.bn0(x)
            x = self.conv1(x)
            x = self.bn1(x)

            if self.high_pass:
                xh = self.conv_fh(x)
                xh = self.bn_fh(xh)

                xv = self.conv_fv(x)
                xv = self.bn_fv(xv)

                x = torch.add(xv, xh)

            else:
                x = self.conv_f(x)
                x = self.bn_f(x)
                                            
            return x
    
    
class SharedEncoder(nn.Module):
    def __init__(self, input_dims, in_channels, out_channels, channels_down, skip_channels = 8):
        super(SharedEncoder, self).__init__()
        
        self.down1 = DownBlock(in_channels, channels_down[0], stride = (2, 2, 2))
        self.down2 = DownBlock(channels_down[0], channels_down[1], stride = (2, 2, 2))
        self.down3 = DownBlock(channels_down[1], channels_down[2], stride = (1, 1, 1))
        self.down4 = DownBlock(channels_down[2], channels_down[3], stride = (1, 1, 1))
        self.down5 = DownBlock(channels_down[3], channels_down[4], stride = (1, 1, 1))

        self.skip1 = SkipBlock(in_channels, skip_channels)
        self.skip2 = SkipBlock(channels_down[0], skip_channels)
        self.skip3 = SkipBlock(channels_down[1], skip_channels)
        self.skip4 = SkipBlock(channels_down[2], skip_channels)
        self.skip5 = SkipBlock(channels_down[3], skip_channels)
                               
    def forward(self, x):
        s1 = self.skip1(x)
        x = self.down1(x)
        s2 = self.skip2(x)
        x = self.down2(x)
        s3 = self.skip3(x)
        x = self.down3(x)
        s4 = self.skip4(x)
        x = self.down4(x)
        s5 = self.skip5(x)
        x = self.down5(x)

        return x, s1, s2, s3, s4, s5
    

class DecoderBranch(nn.Module):
    def __init__(self, input_dims, in_channels, out_channels, channels_down, channels_up, 
                 sigmoid_min, sigmoid_max, high_pass, skip_channels = 8):
        super(DecoderBranch, self).__init__()
        
        self.sigmoid_min = sigmoid_min
        self.sigmoid_max = sigmoid_max

        self.up1 = UpBlock(channels_down[4] + skip_channels, channels_up[4], scale_factor = (1, 1, 1))
        self.up2 = UpBlock(channels_up[4] + skip_channels, channels_up[3], scale_factor = (1, 1, 1))
        self.up3 = UpBlock(channels_up[3] + skip_channels, channels_up[2], scale_factor = (1, 1, 1))
        self.up4 = UpBlock(channels_up[2] + skip_channels, channels_up[1], scale_factor = (2, 2, 2))
        self.up5 = UpBlock(channels_up[1] + skip_channels, channels_up[0], high_pass, scale_factor = (2, 2, 2), 
                           learnable_weights = False)
                
        self.conv = nn.Conv3d(channels_up[0], out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        
    def forward(self, x, s1, s2, s3, s4, s5):
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        
        x = self.conv(x)
        
        x = torch.sigmoid(x) * (self.sigmoid_max - self.sigmoid_min) + self.sigmoid_min
        
        return x

    
class noise_net(nn.Module):
    def __init__(self, input_dims):
        super(noise_net, self).__init__()
        self.input_dims = input_dims
        self.input_noise = torch.nn.Parameter(0.1 * torch.rand(self.input_dims)).type(dtype)
        
    def forward(self):
        return self.input_noise


class model_parallel(nn.Module):
    def __init__(self, input_dims, in_channels, out_channels, channels_down, channels_up_L, channels_up_H,
                 sigmoid_min_L, sigmoid_max_L, sigmoid_min_H, sigmoid_max_H):
        super(model_parallel, self).__init__()
        
        self.enc = SharedEncoder(input_dims[2::], in_channels, out_channels, channels_down).type(dtype).cuda(0)
        
#         self.dec_L = DecoderBranch(input_dims[2::], in_channels, out_channels, channels_down, channels_up_L, 
#                                    sigmoid_min_L, sigmoid_max_L, high_pass = False).type(dtype).cuda(0)      
        self.dec_H = DecoderBranch(input_dims[2::], in_channels, out_channels, channels_down, channels_up_H, 
                                   sigmoid_min_H, sigmoid_max_H, high_pass = False).type(dtype).cuda(1)
        
    def forward(self, z):
        z = z.cuda(0)
        q, s1, s2, s3, s4, s5 = self.enc(z)
#         x_L = self.dec_L(q, s1, s2, s3, s4, s5)
        x_H = self.dec_H(q.cuda(1), s1.cuda(1), s2.cuda(1), s3.cuda(1), s4.cuda(1), s5.cuda(1))

        return x_H


### LOAD DATA
with hp.File('../source/lamino_proj_sinogram_matching_11052021_atcl_2208.h5', 'r') as ee:
    fbp_nit_100 = ee['recon'][()]
    sinogram_shifted = ee['sinogram_shifted'][()]   
    
with hp.File('../source/angles_information.h5', 'r') as ee:
    angles = np.squeeze(ee['angles'][()], axis = 0)

with hp.File('../source/recon_FBP_CoR_offset144_missingConeFilled.mat', 'r') as ee:
    rec_2000_angles = ee['rec'][()]
    
_rec_2000_angles = np.transpose(rec_2000_angles, (2, 1, 0))
_rec_2000_angles = _rec_2000_angles[896 - 160 : 896 + 160, 896 - 160 : 896 + 160, :]
_rec_2000_angles = _rec_2000_angles[160 - 80 - 6 : 160 + 80 - 6, 160 - 80 - 11 : 160 + 80 - 11, 133 : 277]

ee = sio.loadmat('../source/mask_DIP_12122021_autoencoder_skip_intensity_256px.mat')
mask = ee['mask']

fbp_nit_100 = np.transpose(fbp_nit_100, (2, 1, 0))
rec = fbp_nit_100[1104 - 160 : 1104 + 160, 1104 - 160 : 1104 + 160, :]
rec = rec[160 - 80 : 160 + 80, 160 - 80 : 160 + 80, 109 : 253]

sinogram_shifted = np.transpose(sinogram_shifted, (2, 1, 0))
offset_x = 59 + 29
offset_y = 126
exp_sino = sinogram_shifted[480 - 128 + offset_x : 480 + 128 + offset_x, 
                            704 - 128 + offset_y : 704 + 128 + offset_y, :]

del fbp_nit_100, sinogram_shifted
torch.cuda.empty_cache()

print(rec.shape)
print(exp_sino.shape)
print(angles.shape)
print(mask.shape)
print(_rec_2000_angles.shape)


### TRAINING
EPOCHS = 2000
LR = 2e-4

Ny_obj = 160
Nx_obj = 160
Nz_obj = 144
channels_down = [32, 64, 96, 96, 96]
channels_up_L = [24, 48, 72, 72, 72]
channels_up_H = [32, 64, 96, 96, 96]
in_channels = 48
out_channels = 1
input_dims = (1, in_channels, Ny_obj, Nx_obj, Nz_obj)

sigmoid_min_L = -0.015
sigmoid_max_L = 0.015
sigmoid_min_H = -0.03
sigmoid_max_H = 0.03

losses_pred_exp = np.empty(shape = (1 + EPOCHS, ))
losses_pred_exp[:] = np.NaN
param_L_list = np.empty(shape = (1 + EPOCHS, ))
param_L_list[:] = np.NaN
param_H_list = np.empty(shape = (1 + EPOCHS, ))
param_H_list[:] = np.NaN
    
    
input_noise_net = noise_net(input_dims)
net = model_parallel(input_dims, in_channels, out_channels, channels_down, channels_up_L, channels_up_H,
                     sigmoid_min_L, sigmoid_max_L, sigmoid_min_H, sigmoid_max_H)


## Weights initialization
def weights_init_L(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight.data, gain = 0.2)
        
        if m.bias is not None:
             nn.init.constant_(m.bias.data, 0)
                
    elif isinstance(m, nn.BatchNorm3d) and m.weight is not None:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        
def weights_init_H(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight.data, gain = 0.2)
        
        if m.bias is not None:
             nn.init.constant_(m.bias.data, 0)
                
    elif isinstance(m, nn.BatchNorm3d) and m.weight is not None:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        
net.enc.apply(weights_init_L)
net.dec_H.apply(weights_init_H)


def lp_hp_weights(L = 3):
    A = np.zeros((L, L))
    A[int((L-1)/2), int((L-1)/2)] = 1
    
    w_lp_np = np.ones((L, L)) / L**2
    w_hp_np = A - w_lp_np

    w_lp = torch.cuda.FloatTensor(w_lp_np).cuda(0)
    w_hp = torch.cuda.FloatTensor(w_hp_np).cuda(1)
    
    return w_lp, w_hp

_, _ = lp_hp_weights(L = 3)


def repeat_weights(w, sz1, sz2):
    return w.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0).unsqueeze(0).repeat(sz1, sz2, 1, 1, 1)


def train(net, input_noise_net, input_dims, 
          exp_sino, mask, x_true, 
          gamma = 0.5, step_size = 200, num_iter = 3000, LR = 2e-4):
    
    mse = torch.nn.MSELoss().type(dtype)
       
    net_params = list(net.parameters()) + list(input_noise_net.parameters())
    optimizer = torch.optim.Adam(net_params, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = gamma, step_size = step_size)

    for i in range(1, 1 + num_iter):
        if i % 2 == 0:
            RANGE = [int(i) for i in np.floor(np.linspace(0, 250, 125, endpoint = False))]
        else:
            RANGE = [int(i + 1) for i in np.floor(np.linspace(0, 250, 125, endpoint = False))]
            
        y_L = exp_sino[:, :, RANGE] * mask[:, :, RANGE]
        y_H = wavelet_sino_hp(y_L)
        
        y_H_torch = np_to_torch(y_H).type(dtype).cuda(1)

        optimizer.zero_grad()
        for param in net_params:
            param.grad = None

        net_input = input_noise_net()
        out_H = net(net_input) 
        H_out_H = forward_operator(F.pad(out_H, (60, 52, 48, 48, 48, 48)), RANGE, 1)
        H_out_H_sum = torch.sum(H_out_H).cpu().detach()
        
        loss_y_H = mse(H_out_H, y_H_torch.cuda(1))
        total_loss = const_H * loss_y_H.cuda(0)
        total_loss += adaptive_tv_l1_loss_v1(out_H.cuda(0).squeeze(0).squeeze(0), 
                                             w_L = 3e-6, 
                                             w_M = 3e-8,
                                             w_R = 3e-8, 
                                             separate_at_index_L = 100,
                                             separate_at_index_R = 121)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses_pred_exp[i] = total_loss
        
        # Plot and save
        if i % 50 == 0:
            sio.savemat('../rec_6/DIP_PyXL_12092022_H/i_' + str(i) + '_250_angles_5.mat',
                        mdict = {'x_H': out_H.cpu().detach().numpy()})

        if (i % 25 == 0 and i <= 100) or (i % 50 == 0 and i > 100):
            print('Iteration %04d/%04d Loss %f' % (i, num_iter, total_loss.item()), end='\n')
        

        
## Training main loop
train(net = net, input_noise_net = input_noise_net, input_dims = input_dims,  
      exp_sino = exp_sino, mask = mask, x_true = np.expand_dims(np.expand_dims(_rec_2000_angles, 0), 0), 
      gamma = 0.5, step_size = 1000, num_iter = EPOCHS, LR = LR)

