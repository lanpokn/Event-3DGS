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
import torchgeometry as tgm
def rgb_to_grayscale(image):
    # Convert RGB image to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
    # torch is rgb, opencv is bgr
    # grayscale_image = image[0, :, :]
    return grayscale_image.unsqueeze(0)  # Add channel dimension
def rgb_to_LUVscale(image):
    grayscale_image = 0.4124 * image[0, :, :] + 0.35758 * image[1, :, :] + 0.1804 * image[2, :, :]
    grayscale_image = grayscale_image.unsqueeze(0)
    #can't use pow, bad in gradient
    return grayscale_image
def rgb_to_QEscale(image):
    grayscale_image = 0.4124 * image[0, :, :] + 0.35758 * image[1, :, :] + 0.1804 * image[2, :, :]
    grayscale_image = grayscale_image.unsqueeze(0)
    #can't use pow, bad in gradient
    return grayscale_image
def l1_loss_gray(network_output, gt):
    # Convert RGB images to grayscale
    if network_output.size(-3) == 3:
        network_output_gray = rgb_to_grayscale(network_output)
    if gt.size(-3) == 3:    
        gt_gray = rgb_to_grayscale(gt)
        return torch.abs((network_output_gray - gt_gray)).mean()

    return torch.abs((network_output - gt)).mean()
def l1_loss_gray_event(network_output, gt):
    # Convert RGB images to grayscale
    if network_output.size(-3) == 3:
        network_output = rgb_to_grayscale(network_output)
    if gt.size(-3) == 3:    
        gt = rgb_to_grayscale(gt)
    
    # 计算绝对值差值
    # abs_diff = torch.abs(network_output - gt)
    # 在gt为0处应用ReLU函数
    # loss = torch.where(gt == 0, torch.relu(abs_diff - 1), abs_diff)
    # loss = torch.where(abs(gt) < 0.01, abs_diff, 0)
    # loss = torch.where(abs(gt) < 0.01, 0, abs_diff)
    # loss = torch.where(abs(gt) < 0.01, abs_diff, torch.relu(abs_diff - 0.2)) ##this is right, have improment!!!

    abs_diff_1 = torch.abs(network_output - gt-0.5)
    abs_diff_2= torch.abs(gt - network_output-0.5)
    # loss = torch.relu(abs_diff - 0.5)#gt = 3 means 4>nt>3 , gt = -3 means -4<nt<-3, nt-gt
    #BELOW is RIGHT
    loss = torch.where(gt>0, torch.relu(abs_diff_1-0.5), torch.relu(abs_diff_2 - 0.5))
    # 计算平均损失
    return loss.mean()
def l1_filter_loss_gray_event(network_output, gt):
    # Convert RGB images to grayscale
    if network_output.size(-3) == 3:
        network_output = rgb_to_grayscale(network_output)
    if gt.size(-3) == 3:    
        gt = rgb_to_grayscale(gt)
    
    # #这真的对吗，这实现了个啥
    # # 计算绝对值差值
    # abs_diff = torch.abs(network_output - gt)
    # # 计算九个像素中的最小距离
    # min_dist = F.avg_pool2d(network_output, kernel_size=3, stride=1, padding=1)  # 使用均值池化实现获取九个像素的平均值
    # min_dist = torch.abs(min_dist - gt)  # 计算每个像素与 gt 之间的绝对距离
    # min_dist = torch.min(min_dist.view(min_dist.size(0), -1), dim=1).values  # 获取九个像素中的最小距离
    
    # # 在gt不为0处应用ReLU函数
    # loss = torch.where(gt == 0, torch.relu(abs_diff - 1), min_dist)

    # 计算绝对值差值
    abs_diff = torch.abs(network_output - gt)

    # 使用卷积操作获取邻近九个像素的绝对差值
    kernel = torch.ones(1, 1, 1, 2).to(network_output.device)
    l1_diff = torch.nn.functional.conv2d(abs_diff, kernel, padding=1)/2

    # 获取最小的差值
    mindist, _ = torch.min(l1_diff, dim=1)
    mindist = mindist.squeeze(0)
    # return mindist.squeeze(0).mean()  # 去除扩展的维度
    # loss = torch.where(gt == 0, torch.relu(abs_diff - 1), mindist)
    loss = mindist
    # 计算平均损失
    return loss.mean()
def chamfer_loss(img_list1, img_list2):
    # Convert images to coordinate lists
    def calculate_center_of_mass(img_list1_tensor):
        """
        计算图像的质心

        参数：
        img_list1_tensor: 输入图像的PyTorch张量表示

        返回：
        center_of_mass_x: 质心的 x 坐标
        center_of_mass_y: 质心的 y 坐标
        """
        # 创建坐标网格
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i_coords, j_coords = torch.meshgrid(torch.arange(img_list1_tensor.shape[0], device=device), torch.arange(img_list1_tensor.shape[1], device=device))
        # 计算加权坐标总和和权重总和
        sum_x = torch.sum(i_coords * img_list1_tensor)
        sum_y = torch.sum(j_coords * img_list1_tensor)
        total_weight = torch.sum(img_list1_tensor)

        # 计算质心
        center_of_mass_x = sum_x / total_weight
        center_of_mass_y = sum_y / total_weight
        
        return center_of_mass_x.item(), center_of_mass_y.item()
    def image_to_coordinate_list(image):
        # Convert image to coordinate list of non-zero points
        # non_zero_indices = torch.nonzero(image, as_tuple=False)
        non_zero_indices = torch.nonzero(torch.abs(image) > 0.85, as_tuple=False)
        return non_zero_indices.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_list1.size(-3) == 3:
        img_list1 = 0.299 * img_list1[0, :, :] + 0.587 * img_list1[1, :, :] + 0.114 * img_list1[2, :, :]
    else:
        img_list1 = img_list1[0, :, :]
    if img_list2.size(-3) == 3:    
        img_list2 = 0.299 * img_list2[0, :, :] + 0.587 * img_list2[1, :, :] + 0.114 * img_list2[2, :, :]
    else:
        img_list2 = img_list2[0, :, :]

    #x' = sigma i*img_list1(i,j)/img_list1(i,j)
    x1,y1 = calculate_center_of_mass(img_list1)
    x2,y2 = calculate_center_of_mass(img_list1)
    img_list1 = image_to_coordinate_list(img_list1).squeeze(1)
    img_list2 = image_to_coordinate_list(img_list2).squeeze(1)
    # 将质心坐标转换为张量并将其移动到计算设备上
    center_of_mass1 = torch.tensor([x1, y1], dtype=img_list1.dtype, device=device)
    center_of_mass2 = torch.tensor([x2, y2], dtype=img_list2.dtype, device=device)

    # 将质心从图像列表中减去
    img_list1 = img_list1 - center_of_mass1.unsqueeze(1).unsqueeze(1)
    img_list2 = img_list2 - center_of_mass2.unsqueeze(1).unsqueeze(1)
    # 将 img_list1 中的值除以5，并且删除重复元素
    # 将 img_list1 中的值除以5，并且删除重复元素
    # img_list1 = torch.unique(img_list1 / 10, dim=0)

    # # 将 img_list2 中的值除以5，并且删除重复元素
    # img_list2 = torch.unique(img_list2 / 10, dim=0)
    # Compute pairwise distances
    dists = torch.cdist(img_list1, img_list2, p=2)
    
    # Find nearest neighbors
    min_dist_1_to_2, _ = torch.min(dists, dim=1)
    # min_dist_2_to_1, _ = torch.min(dists, dim=0)
    
    # Compute average distance
    avg_dist = torch.mean(min_dist_1_to_2)
    
    return avg_dist
def differentialable_threld(x,C=0.3,e=0.00001,w = 10):
    torch.sign(x)/(1 + torch.exp(w*(C - torch.abs(x))))

def differentialable_event_simu(image,image_next,gt_image,C=0.3):
    #return event number! img_diff can be negtive
    # img1 = rgb_to_grayscale(image)
    # img2 = rgb_to_grayscale(image_next)
    img1 = rgb_to_LUVscale(image)
    img2 = rgb_to_LUVscale(image_next)
    # torchvision.utils.save_image(img1, "img1.png")
    # torchvision.utils.save_image(img2, "img2.png")
    #total physical, but not work
    epsilon = 1e-8  # avoid dividing 0
    # C=10
    img_diff = (torch.log(img2+epsilon) - torch.log(img1+epsilon))/C

    # threshold = 0
    # img_diff = torch.where(torch.abs(img_diff) < C*0.8 , torch.tensor(0.0), img_diff)
    return img_diff


def Normalize_event_frame(gt_image):
    #torch can only be from 0 to 1
    #if both -1,1 is given one of them will become 0
    #TODO, no 1 only 0.5 and -1, why?
    # event_image = torch.full_like(gt_image[0:1, :, :], 0.5)
    # 计算新的 event_image
    if gt_image.shape[0] == 3:
        event_image = (gt_image[0,:, :] - gt_image[2,:, :]) / (10 / 255)
        return event_image.unsqueeze(0)
    # if gt_image.shape[2] == 3:
    #     event_image = (gt_image[:, :,0] - gt_image[:, :,2])
    #     event_image = event_image.transpose(1, 2)
    #     event_image = event_image.transpose(0, 1)
    #     return event_image.unsqueeze(2)

    # assert gt_image.shape[0] == 3, "Input image must have three channels"


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
def l1_loss_event_new(network_output, gt, tolerance=0.2):
    # 计算差的绝对值
    abs_diff = torch.abs(network_output - gt)
    abs_diff = abs_diff
    # torchvision.utils.save_image(abs_diff, "abs_diff.png")
    # 计算每个像素上的损失
    pixel_loss = torch.where(abs_diff < tolerance, torch.tensor(0.0), ((abs_diff - tolerance)*1000).pow(2))

    # 计算平均损失
    average_loss = pixel_loss.mean()

    return average_loss
def l1_loss_event(network_output, gt, weight=10000):
    def grayscale_to_pointcloud(image,network,device):


        x, y = image.shape[1], image.shape[2]

        # 生成坐标
        indices = torch.arange(x * y, device=device)
        x_coords = (indices // y).float() 
        y_coords = (indices % y).float()

        # 生成pointcloud
        pointcloud = torch.stack([x_coords, y_coords, image.view(-1)], dim=1)
        pointcloud_work = torch.stack([x_coords, y_coords, network.view(-1)], dim=1)
        # 筛选值大于 threshold 的点
        mask_gt = pointcloud[:, 2] > 0.9
        pointcloud_gt = pointcloud[mask_gt]
        pointcloud_network_gt = pointcloud_work[mask_gt]
        mask_lt = pointcloud[:, 2] < -0.9
        pointcloud_lt = pointcloud[mask_lt]
        pointcloud_network_lt = pointcloud_work[mask_lt]

        return pointcloud_gt, pointcloud_lt,pointcloud_network_gt,pointcloud_network_lt
    #this is wrong,but have a good result, why??
    # def grayscale_to_pointcloud(image, device):
    #     _, h, w = image.shape
    #     y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    #     x_coords = x_coords.flatten()
    #     y_coords = y_coords.flatten()
    #     z_coords = image.flatten()

    #     # 筛选值大于 threshold 的点
    #     mask_gt = z_coords > 0.9
    #     pointcloud_gt = torch.stack([x_coords[mask_gt], y_coords[mask_gt], z_coords[mask_gt]], dim=1)

    #     # 筛选值小于 threshold 的点
    #     mask_lt = z_coords < 0.1
    #     pointcloud_lt = torch.stack([x_coords[mask_lt], y_coords[mask_lt], z_coords[mask_lt]], dim=1)

    #     return pointcloud_gt, pointcloud_lt
    def sample_points(pointcloud, num_points):
        if pointcloud.shape[0] <= num_points:
            return pointcloud
        else:
            indices = torch.randperm(pointcloud.shape[0])[:num_points]
            return pointcloud[indices]
    def process_pointcloud(pointcloud_gt, pointcloud_lt, max_points=3000):
        pointcloud_gt_processed = sample_points(pointcloud_gt, max_points)
        pointcloud_lt_processed = sample_points(pointcloud_lt, max_points)
        return pointcloud_gt_processed, pointcloud_lt_processed
    # n_gt,n_lt = grayscale_to_pointcloud(network_output, device=network_output.device)
    g_gt,g_lt,n_gt,n_lt = grayscale_to_pointcloud(gt,network_output,device=gt.device)
    n_gt, n_lt = process_pointcloud(n_gt, n_lt, max_points=9000)
    g_gt, g_lt = process_pointcloud(g_gt, g_lt, max_points=9000)

    # 计算两个点云之间的距离矩阵
    #TODO if no point, skip and give default value
    #give a max number  to avoid exceed GPU
    # torchvision.utils.save_image(gt, "gt.png")
    if n_gt.numel() == 0 or g_gt.numel() == 0 or g_lt.numel() == 0 or n_lt.numel() == 0:
        return l1_loss(network_output, gt)
    distances = torch.cdist(n_gt[:, :2], g_gt[:, :2])
    _, indices = torch.min(distances, dim=1)
    nearest_points = g_gt[indices, :]
    distances1 = torch.norm(n_gt[:, :2] - nearest_points[:, :2], dim=1).mean()

    distances = torch.cdist(n_lt[:, :2], g_lt[:, :2])
    _, indices = torch.min(distances, dim=1)
    nearest_points = g_lt[indices, :]
    distances2 = torch.norm(n_lt[:, :2] - nearest_points[:, :2], dim=1).mean()

    return distances1+distances2
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
    else:
        img1_gray = img1
    if img2.size(-3) == 3:    
        img2_gray = rgb_to_grayscale(img2)
    else:
        img2_gray = img2

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

