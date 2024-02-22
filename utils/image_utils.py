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
import lpips
import sys
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def LPIPS(img1, img2):
    # Move img1 and img2 to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Load LPIPS model on GPU
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = None
    sys.stderr = None

    lpips_model = lpips.LPIPS(net='alex').to(device)

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Compute LPIPS similarity
    return lpips_model(img1, img2)