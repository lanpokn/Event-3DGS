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
from torchvision.transforms import ToPILImage
import cv2
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_point,render_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from Event_sensor.event_tools import *
import copy
from Event_sensor.src.event_buffer import EventBuffer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import json
import math

def Nlerp(a1,a2,alpha):
    return alpha * a1 + (1 - alpha) *a2
#TODO problem of discontinulity not solved
def Slerp(a1,a2,alpha):
    cosfi = a1[1]*a2[1]+a1[2]*a2[2]+a1[3]*a2[3]
    #to fix the discontinulity
    if(abs(a1[1]-a2[1])>0.5):
        a2 = -a2
    fi = math.acos(cosfi)
    ret = math.sin(fi*(1-alpha))*a1/(math.sin(fi)+0.000001) + math.sin(fi*alpha)*a2/(math.sin(fi)+0.000001)
    return ret
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # Define paths for rendered images and ground truth
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    """
    Render sets of images for training and testing using the specified parameters.

    Args:
        dataset (ModelParams): Parameters of the model and dataset.
        iteration (int): Iteration number for loading the scene.
        pipeline (PipelineParams): Parameters for the rendering pipeline.
        skip_train (bool): Whether to skip rendering images for training.
        skip_test (bool): Whether to skip rendering images for testing.

    Returns:
        None
    """
    #forbidden gradient computation
    with torch.no_grad():
        # Create Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)

        # Load scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Set background color based on dataset
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Render training set if not skipped
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # Render test set if not skipped
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
def rendering_to_cvimg(rendering):
    to_pil = ToPILImage()
    pil_image = to_pil(rendering)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencv_image
def Generate_new_view(view,R,T):
    view_new = copy.deepcopy(view)
    view_new.R = R
    view_new.T = T
    view_new.world_view_transform = torch.tensor(getWorld2View2(R, T, view.trans, view.scale)).transpose(0, 1).cuda()
    view_new.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
    view_new.full_proj_transform = (view_new.world_view_transform.unsqueeze(0).bmm(view_new.projection_matrix.unsqueeze(0))).squeeze(0)
    view_new.camera_center = view_new.world_view_transform.inverse()[3, :3]
    return view_new

#TODO,camera_angle_x and rotation specifyied by command line
#TODO, still hard to get a good transform_matrix, may need to specify it per element
def generate_transforms_json(view_list, file_path_prefix,json_path,camera_angle_x, rotation):
    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }

    for i in range(len(view_list)):
        file_path = f"{file_path_prefix}_{i:05d}"
        #TODO
        transform_matrix = view_list[i].world_view_transform  # Assuming view_temp_list contains the transform_matrix for each frame
        frame_data = {
            "file_path": file_path,
            "rotation": rotation,
            "transform_matrix": transform_matrix.transpose(0, 1).tolist()  # Converting numpy array to list for JSON serialization
        }
        transforms_data["frames"].append(frame_data)

    with open(json_path, "w") as json_file:
        json.dump(transforms_data, json_file, indent=4)

def edge_preserving_interpolation(depth_image, spatial_sigma, depth_sigma):
    # 使用双边滤波器进行边缘保持插值
    depth_image = rendering_to_cvimg(depth_image)
    interpolated_depth = cv2.bilateralFilter(depth_image, d=0, sigmaColor=depth_sigma, sigmaSpace=spatial_sigma)

    return interpolated_depth
def render_set_event(model_path, name, iteration, views, gaussians, pipeline, background,args):
    # Define paths for rendered images and ground truth
    if len(views) == 0:
        return
    maxLoopN = args.maxLoopN
    old_event = args.old_event
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    event_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event")
    event_ac_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event_ac")
    if old_event == True:
        event_old_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event_old")
        makedirs(event_old_path, exist_ok=True)
        img_old_list = []
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(event_path, exist_ok=True)
    makedirs(event_ac_path, exist_ok=True)
    img_list = []
    view_list = []
    interpolation_number = args.interpolationN
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view_list.append(view)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx*interpolation_number+0) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        opencv_image = rendering_to_cvimg(rendering)
        img_list.append(opencv_image)
        if old_event == True:
            opencv_image = rendering_to_cvimg(gt)
            img_old_list.append(opencv_image)


        if idx+1 == len(views):
            break
        if idx>maxLoopN:
            break
        view_next = views[idx+1]
        q_start = rotation_matrix_to_quaternion(view.R)
        q_end = rotation_matrix_to_quaternion(view_next.R)
        T_start = view.T
        T_end = view_next.T
        for i in range(1,interpolation_number):
            alpha = i / interpolation_number  # Linear interpolation parameter
            # TODO , how to get better interpolation
            #now only using Nlerp
            q_temp = Slerp(q_end,q_start,alpha)
            q_temp = q_temp/np.linalg.norm(q_temp)
            R_temp = quaternion_to_rotation_matrix(q_temp)
            # Linear interpolation for translation vectors
            T_temp = Nlerp(T_end,T_start,alpha)
            # Create a temporary view
            view_temp = Generate_new_view(view,R_temp,T_temp)
            view_list.append(view_temp)
            rendering = render(view_temp, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx*interpolation_number+i) + ".png"))
            opencv_image = rendering_to_cvimg(rendering)
            img_list.append(opencv_image)
    ev_full = EventBuffer(1)
    dt = 1000
    generate_transforms_json(view_list,"./train/",os.path.join(model_path, name, "ours_{}".format(iteration),"transforms_train.json"),6911112070083618,0.031415926535897934)
    generate_transforms_json(view_list,"./train_event/",os.path.join(model_path, name, "ours_{}".format(iteration),"transforms_train_event.json"),6911112070083618,0.031415926535897934)
    print("generating events...")
    simulate_event_camera(img_list,ev_full,dt)
    print("saving ...")
    save_event_result(ev_full,event_path)
    # generate_images(event_path,dt,(maxLoopN+1)*interpolation_number,img_list[0].shape[1],img_list[0].shape[0])
    # generate_images_accumu(event_path,dt,(maxLoopN+1)*interpolation_number,img_list[0].shape[1],img_list[0].shape[0])
    generate_images(event_path,dt,(maxLoopN+1)*interpolation_number,img_list[0].shape[1],img_list[0].shape[0])
    generate_images_accumu(event_path,dt,(maxLoopN+1)*interpolation_number,img_list[0].shape[1],img_list[0].shape[0])

    if old_event == True:
        ev_full_old = EventBuffer(1)
        dt_old = 2857*interpolation_number
        print("generating old events...")
        simulate_event_camera(img_old_list,ev_full_old,dt_old)
        print("saving ...")
        save_event_result(ev_full_old,event_old_path)
        generate_images(event_old_path,dt_old,maxLoopN,img_old_list[0].shape[1],img_old_list[0].shape[0])

#it's hard to precisely give a velocity, only use a positively related paramter
#blurrySpeed: frame_per_shutterTime, if 2 then,frame 2'RGB will open at frame 1 and close at frame 3
#assume: time from frame i to frame i+1 is the same because orginal motion is unknown, this is enough for test
#if you want better: specify w and v, and intepolation curve between neighbor R and T
#better not exceed 2    
def render_set_blurry(model_path, name, iteration, views, gaussians, pipeline, background,args):
    # Define paths for rendered images and ground truth
    if len(views) == 0:
        return
    blurrySpeed = args.blurrySpeed
    maxLoopN = args.maxLoopN
    blurry_path = os.path.join(model_path, name, "ours_{}".format(iteration), "blurry")

    makedirs(blurry_path, exist_ok=True)
    img_list = []
    #bigger inter number, better in realism of blurry
    interpolation_number = 18
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    interpolation_N = args.interpolationN
    rendering_list = []
    for idx in tqdm(range(0, len(views), 1), desc="Rendering progress"):
        view = views[idx]
        #skip no blurry img
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        opencv_image = rendering_to_cvimg(rendering)
        img_list.append(opencv_image)
        if idx+1 == len(views):
            break
        if idx>maxLoopN:
            break
        view_next = views[idx+1]
        q_i = rotation_matrix_to_quaternion(view.R)
        q_iplus1 = rotation_matrix_to_quaternion(view_next.R)
        T_i= view.T
        T_iplus1 = view_next.T
        alpha_blurry = 0.5*blurrySpeed
        q_end = Nlerp(q_iplus1,q_i,alpha_blurry)
        q_end = q_end/np.linalg.norm(q_end)
        T_end = Nlerp(T_iplus1,T_i,alpha_blurry)
        #if first img then use twice time after first
        # rendering_list = []
        if idx == 0:
            q_start = q_i
            T_start = T_i
            q_end = Nlerp(q_iplus1,q_i,alpha_blurry*2)
            q_end = q_end/np.linalg.norm(q_end)
            T_end = Nlerp(T_iplus1,T_i,alpha_blurry*2)
        else:
            view_previous = views[idx-1]
            q_iminus1 = rotation_matrix_to_quaternion(view_previous.R)
            T_iminus1 = view_previous.T
            q_start = Nlerp(q_iminus1,q_i,alpha_blurry)
            q_start = q_start/np.linalg.norm(q_start)
            T_start = Nlerp(T_iminus1,T_i,alpha_blurry)
        for i in range(0,interpolation_number+1):
            alpha = i / interpolation_number  # Linear interpolation parameter
            # TODO , how to get better interpolation
            #now only using Nlerp
            q_temp = Nlerp(q_end,q_start,alpha)
            q_temp = q_temp/np.linalg.norm(q_temp)
            R_temp = quaternion_to_rotation_matrix(q_temp)
            # Linear interpolation for translation vectors
            T_temp = Nlerp(T_end,T_start,alpha)
            # Create a temporary view
            view_temp = Generate_new_view(view,R_temp,T_temp)
            rendering = render(view_temp, gaussians, pipeline, background)["render"]
            rendering_list.append(rendering)
    dt = int(interpolation_number/interpolation_N)
    for i in range(0,len(rendering_list)-interpolation_number,dt):
        temp = []
        for j in range(0,interpolation_number):
            temp.append(rendering_list[i+j])
        rendering_tensor = torch.stack(temp)
        average_rendering = torch.mean(rendering_tensor, dim=0)
        torchvision.utils.save_image(average_rendering, os.path.join(blurry_path, '{0:05d}'.format(int(i/3)) + ".png"))
    # generate_transforms_json()

def render_set_point(model_path, name, iteration, views, gaussians, pipeline, background,args):
    # Define paths for rendered images and ground truth
    point_path = os.path.join(model_path, name, "ours_{}".format(iteration), "point")

    makedirs(point_path, exist_ok=True)

    maxLoopN = args.maxLoopN
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx >maxLoopN:
            break
        point_map = render_point(view, gaussians, pipeline, background)
        # Assuming point_map is your tensor
        point_map_float = point_map.clone()  # Create a copy to avoid modifying the original tensor

        # Find the minimum and maximum values excluding inf
        min_val = point_map_float[point_map_float != float('inf')].min()
        max_val = point_map_float[point_map_float != float('inf')].max()

        # Normalize the non-inf values to the range [0, 1)
        point_map_float[point_map_float != float('inf')] = (point_map_float[point_map_float != float('inf')] - min_val) / (max_val - min_val)
        point_map_float[point_map_float == float('inf')] = 100
        save_path = os.path.join(point_path, '{0:05d}_min{1:.4f}_max{2:.4f}.png'.format(idx, min_val.item(), max_val.item()))
        torchvision.utils.save_image(point_map_float,save_path)

## get idea from:
#Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images
#thank you!
def render_set_depth(model_path, name, iteration, views, gaussians, pipeline, background,args):
    # Define paths for rendered images and ground truth
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(depth_path, exist_ok=True)

    maxLoopN = args.maxLoopN
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx >maxLoopN:
            break
        rendering = render_depth(view, gaussians, pipeline, background)["render"]
        rendering = rendering/10
        # gt = view.original_image[0:3, :, :]
        # img = rendering_to_cvimg(rendering)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # 将灰度图像转换为浮点型
        # float_gray_img = gray_img.astype(np.float32)
        # cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"),float_gray_img)
        torchvision.utils.save_image(rendering, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets_mixed(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    """
    Render sets of images for training and testing using the specified parameters.

    Args:
        dataset (ModelParams): Parameters of the model and dataset.
        iteration (int): Iteration number for loading the scene.
        pipeline (PipelineParams): Parameters for the rendering pipeline.
        skip_train (bool): Whether to skip rendering images for training.
        skip_test (bool): Whether to skip rendering images for testing.

    Returns:
        None
    """
    skip_train = args.skip_train
    skip_test = args.skip_test
    blurrySpeed = args.blurrySpeed
    point = args.point
    depth = args.depth
    #forbidden gradient computation
    with torch.no_grad():
        # Create Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)

        # Load scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Set background color based on dataset
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Render training set if not skipped
        if not skip_train:
            if depth:
                render_set_depth(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            if point:
                render_set_point(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            if blurrySpeed > 0:
                render_set_blurry(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            render_set_event(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)

        # Render test set if not skipped
        if not skip_test:
            if depth:
                render_set_depth(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            if point:
                render_set_point(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            if blurrySpeed > 0:
                render_set_blurry(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,args)
            render_set_event(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--maxLoopN", default=-1, type=int)
    parser.add_argument("--old_event", action="store_true")
    parser.add_argument("--blurrySpeed", default=-1, type=float)
    parser.add_argument("--interpolationN", default=3, type=int)
    parser.add_argument("--point", action="store_true")
    parser.add_argument("--depth", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_mixed(model.extract(args), args.iteration, pipeline.extract(args), args)