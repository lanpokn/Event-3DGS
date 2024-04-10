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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim,l1_loss_gray,ssim_gray,differentialable_event_simu,Normalize_event_frame,rgb_to_grayscale
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr,LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from Event_sensor.event_tools import *
import copy
from Event_sensor.src.event_buffer import EventBuffer
import torchvision
from render import Generate_new_view
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import random
def generate_random_integer_nearby(target_integer, range_half_width):
    """
    Generate a random integer nearby the target_integer within the specified range_half_width.
    """
    lower_bound = target_integer - range_half_width
    upper_bound = target_integer + range_half_width
    return random.randint(lower_bound, upper_bound)
class CameraPoseInterpolator:
    def __init__(self, camera_poses,dt):
        self.camera_poses = camera_poses
        self.dt = dt

    def interpolate_pose_at_time(self, t):
        """
        Interpolates camera pose at a given time t.
        more range to avoid out of range
        """
        for idx in range(-10,len(self.camera_poses) + 10):
            if idx <0:
                view = self.camera_poses[0]
                view_next = self.camera_poses[1]
            else:
                view = self.camera_poses[len(self.camera_poses)-1]
                view_next = self.camera_poses[len(self.camera_poses)-2]

            view = self.camera_poses[idx]
            view_next = self.camera_poses[idx + 1]

            if t >= self.dt*idx and t <= self.dt*(idx+1):
                alpha = (t - self.dt*idx) / (self.dt)

                q_start = rotation_matrix_to_quaternion(view.R)
                q_end = rotation_matrix_to_quaternion(view_next.R)
                T_start = view.T
                T_end = view_next.T

                q_temp = Nlerp(q_end, q_start, alpha)
                q_temp /= np.linalg.norm(q_temp)
                R_temp = quaternion_to_rotation_matrix(q_temp)

                T_temp = Nlerp(T_end, T_start, alpha)
                view_temp = Generate_new_view(view,R_temp,T_temp)
                return view_temp
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,event_path):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # #get evennt data
    # if event_path != None:
    #     events_data = EventsData()
    #     events_data.read_IEBCS_events(os.path.join(event_path,"raw.dat"), 10000000)
    #     ev_data = events_data.events[0] 
    if checkpoint:
        (model_params, __) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    #TODO first generate a camera position according to scene.getTrainCameras().copy()
    #use t as indicator
    Interpolator = CameraPoseInterpolator(scene.getTrainCameras(),13513*10)
    # pose = Interpolator.interpolate_pose_at_time(4000)
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    ##may need to multi higher
                    if args.gray == True:
                        net_image  = 0.299 * net_image[0, :, :] + 0.587 * net_image[1, :, :] + 0.114 * net_image[2, :, :]
                        net_image = torch.stack([net_image, net_image, net_image], dim=0)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()
        # Pick a random Camera
        # only copy in first iteration
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        #TODO:Check whether 3DGS's ssim is right
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        bg_gray = 0.4
                 
        if not viewpoint_stack:
            viewpoint_stack = scene.getTestCameras().copy()
            if args.deblur == True:
                viewpoint_blurry_stack = scene.getBlurryCameras().copy()
            if args.event == True:
                viewpoint_Event_stack = scene.getEventCameras().copy()
        ssim_test = 0.0
        psnr_test = 0.0
        LPiPS_test = 0.0
        index_list = [5,25,45,65,85]
        #index_list = [5,25]
        #TODO:Check whether 3DGS's ssim is right
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        bg_gray = 0.73
        for index in index_list:
            #test 0, 5, 10 ...render(viewpoint_cam_pre, gaussians, pipe, bg)
            viewpoint = viewpoint_stack[index]
            #output gray graph
            if args.e2vid == True:
                image = scene.getTrainCameras().copy()[index].original_image.to("cuda")
                gt_image = viewpoint_stack[index].original_image.to("cuda")
                image = torch.clamp(image, 0.0, 1.0)
                gt_image = torch.clamp(gt_image, 0.0, 1.0)
                gt_image = rgb_to_grayscale(gt_image)
            else:
                image = render(viewpoint, gaussians, pipe, bg)["render"]
                gt_image = viewpoint_stack[index].original_image.to("cuda")
                image = torch.clamp(image, 0.0, 1.0)
                gt_image = torch.clamp(gt_image, 0.0, 1.0)
                image = rgb_to_grayscale(image)
                gt_image = rgb_to_grayscale(gt_image)
                # only in mic
                # gt_image = torch.where(abs(gt_image -166/255) > 0.0001, gt_image, bg_gray)
            # image = image/torch.mean(image)
            # gt_image = gt_image/torch.mean(gt_image)
            torchvision.utils.save_image(image,'images/sim_{:05d}.{}'.format(index, "png"))
            torchvision.utils.save_image(gt_image,'images/real_{:05d}.{}'.format(index, "png"))
            ssim_test += ssim(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            LPiPS_test+= LPIPS(image, gt_image).mean().double()
        psnr_test /= len(index_list)
        ssim_test /= len(index_list)
        LPiPS_test /= len(index_list)            
        print("\n[SSIM {} PSNR {} LPiPS {}".format(ssim_test, psnr_test,LPiPS_test))
        return 
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
        #TODO ssim and LPIPS
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    #test 0, 5, 10 ...
                    if idx%5 == 0:
                        continue
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_999,4000,5999,6999,7999])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1])#need --eval,whether --eval matters train a lot , do not turn it in training!
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[999,1999,2999,3999,5999,6999,7999,8999,9999,10999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[999,1999,2999,3999,5999,6999,7999,8999,9999,10999])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dat", type=str, default = None)
    parser.add_argument("--e2vid",  action="store_true")
    parser.add_argument("--blur",  action="store_true")

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.dat)

    # All done
    print("\nTraining complete.")
