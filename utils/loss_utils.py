#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision
def rgb_to_grayscale(image):
    # Convert RGB image to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
    # torch is rgb, opencv is bgr
    # grayscale_image = image[0, :, :]
    return grayscale_image.unsqueeze(0)  # Add channel dimension

def l1_loss_gray(network_output, gt):
    # Convert RGB images to grayscale
    if network_output.size(-3) == 3:
        network_output_gray = rgb_to_grayscale(network_output)
    if gt.size(-3) == 3:    
        gt_gray = rgb_to_grayscale(gt)

    return torch.abs((network_output_gray - gt_gray)).mean()
def differentialable_threld(x,C=0.3,e=0.00001,w = 10):
    torch.sign(x)/(1 + torch.exp(w*(C - torch.abs(x))))

def differentialable_event_simu(image,image_next):
    img1 = rgb_to_grayscale(image)
    img2 = rgb_to_grayscale(image_next)
    torchvision.utils.save_image(img1, "img1.png")
    torchvision.utils.save_image(img2, "img2.png")
    # ##total physical, but not work
    # epsilon = 1e-8  # avoid dividing 0
    # img_diff = torch.log(img2) - torch.log(img1)
    # C=0.3
    # w=10
    # factor1 = torch.sign(img_diff)
    # factor2 = (1 + torch.exp(w * (C - torch.abs(img_diff))))
    # result = (factor1 / factor2 + 1)/2
    # torchvision.utils.save_image(result, "test.png")

    #another way
    epsilon = 1e-8  # avoid dividing 0
    img_diff =(img2) - (img1)
    C=0.3
    w=10
    factor1 = torch.sign(img_diff)
    factor2 = (1 + torch.exp(w * (C - torch.abs(img_diff))))
    result = (factor1 / factor2 + 1)/2
    torchvision.utils.save_image(result, "test.png")

    return result

def Normalize_event_frame(gt_image):
    #torch can only be from 0 to 1
    #if both -1,1 is given one of them will become 0
    event_image = torch.full_like(gt_image[0:1, :, :], 0.5)

    # positive
    condition_1 = torch.logical_and(gt_image[0, :, :] > 0.1, gt_image[0, :, :] < 0.9)
    #negtive
    condition_2 = torch.logical_and(gt_image[2, :, :] > 0.1, gt_image[2, :, :] < 0.9)

    event_image[0,condition_1] = 1
    event_image[0,condition_2] = 0
    # torchvision.utils.save_image(event_image, "test.png")
    return event_image

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def ssim_gray(img1, img2, window_size=11, size_average=True):
    if img1.size(-3) == 3:
        img1_gray = rgb_to_grayscale(img1)
    if img2.size(-3) == 3:    
        img2_gray = rgb_to_grayscale(img2)

    channel = img1_gray.size(-3)
    window = create_window(window_size, channel)
    # Convert RGB images to grayscale
    if img1_gray.is_cuda:
        window = window.cuda(img1_gray.get_device())
    window = window.type_as(img1_gray)

    return _ssim(img1_gray, img2_gray, window, window_size, channel, size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

