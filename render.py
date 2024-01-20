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
def render_set_event(model_path, name, iteration, views, gaussians, pipeline, background):
    # Define paths for rendered images and ground truth
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    event_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(event_path, exist_ok=True)
    img_list = []
    gt = view.original_image[0:3, :, :]
    rendering = render(view, gaussians, pipeline, background)["render"]
    interpolation_number = 4
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        img_list.append(rendering)
        if idx+1 == len(views):
            break
        view_next = views(idx+1)
        q_start = rotation_matrix_to_quaternion(view.R)
        q_end = rotation_matrix_to_quaternion(view_next.R)
        T_start = view.T
        T_end = view_next.T
        for i in range(1,interpolation_number):
            alpha = i / interpolation_number  # Linear interpolation parameter
            # Linear interpolation for quaternions
            q_temp = torch.slerp(torch.Tensor(q_start), torch.Tensor(q_end), alpha)
            # Linear interpolation for translation vectors
            T_temp = alpha * T_end + (1 - alpha) * T_start
            # Create a temporary view
            view_temp = copy.deepcopy(view)
            view_temp.R = quaternion_to_rotation_matrix(q_temp)
            view_temp.T = T_temp
            rendering = render(view_temp, gaussians, pipeline, background)["render"]
            img_list.append(rendering)
        ev_full = EventBuffer(1)
        dt = 2857
        simulate_event_camera(img_list,ev_full,2857)
        save_event_result(ev_full,event_path)
        generate_images(event_path,dt*interpolation_number,total_dt_nums=300)
        

def render_sets_event(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
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
            render_set_event(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # Render test set if not skipped
        if not skip_test:
            render_set_event(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_event(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)