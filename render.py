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
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from Event_sensor.event_tools import *
import copy
from Event_sensor.src.event_buffer import EventBuffer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
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

    

def render_set_event(model_path, name, iteration, views, gaussians, pipeline, background,maxLoopN):
    # Define paths for rendered images and ground truth
    if len(views) == 0:
        return
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    event_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event")
    event_old_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event_old")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(event_path, exist_ok=True)
    makedirs(event_old_path, exist_ok=True)
    img_list = []
    img_old_list = []
    interpolation_number = 3
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx*interpolation_number+0) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        opencv_image = rendering_to_cvimg(rendering)
        img_list.append(opencv_image)
        
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
            q_temp = (1 - alpha) * q_start + alpha * q_end
            q_temp = q_temp/np.linalg.norm(q_temp)
            R_temp = quaternion_to_rotation_matrix(q_temp)
            # Linear interpolation for translation vectors
            T_temp = alpha * T_end + (1 - alpha) * T_start
            # Create a temporary view
            view_temp = Generate_new_view(view,R_temp,T_temp)
            rendering = render(view_temp, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx*interpolation_number+i) + "po.png"))
            opencv_image = rendering_to_cvimg(rendering)
            img_list.append(opencv_image)
    ev_full = EventBuffer(1)
    dt = 2857
    print("generating events...")
    simulate_event_camera(img_list,ev_full,2857)
    print("saving ...")
    save_event_result(ev_full,event_path)
    generate_images(event_path,dt,maxLoopN*interpolation_number,img_list[0].shape[1],img_list[0].shape[0])

    ev_full_old = EventBuffer(1)
    dt_old = 2857*interpolation_number
    print("generating old events...")
    simulate_event_camera(img_old_list,ev_full_old,dt_old)
    print("saving ...")
    save_event_result(ev_full_old,event_old_path)
    generate_images(event_old_path,dt_old,maxLoopN,img_old_list[0].shape[1],img_old_list[0].shape[0])
        

def render_sets_event(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,maxLoopN:int):
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
            render_set_event(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,maxLoopN)

        # Render test set if not skipped
        if not skip_test:
            render_set_event(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,maxLoopN)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--maxLoopN", default=-1, type=int)
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_event(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.maxLoopN)