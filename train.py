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
from utils.image_utils import psnr
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
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        # only copy in first iteration
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if args.deblur == True:
                viewpoint_blurry_stack = scene.getBlurryCameras().copy()
            if args.event == True:
                viewpoint_Event_stack = scene.getEventCameras().copy()

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if args.event == True:
            #TODO, in fact, it's better to be -2
            #index = randint(0, len(viewpoint_stack)-2)
            index = randint(1, len(viewpoint_stack)-3)
            # if index == 48 or index == 49:
            #     index=3
        else:
            # index = randint(0, len(viewpoint_stack)-1)
            index = randint(1, len(viewpoint_stack)-2)

        #colmap do not support additional test images
        #thus we use manully test
        #test on 5,25,45,65,85
        if args.event == True or args.gray == True:
            if index == 5 or index == 25 or index == 45 or index == 65 or index == 85:
                index=index -1
        
        # viewpoint_cam = viewpoint_stack.pop(index)
        #delete some bad img
            
        # index = 1
        viewpoint_cam = viewpoint_stack[index]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # torchvision.utils.save_image(image, "test.png")

        # Loss
        # TODO, change to gray
        # event: +1,-1,0, just to be a type of gray and need an initial img
        # not that hard,just use a stack to locate img before
        # thus,first event frame t =i corre to 3DGS frame i-dt and i+dt,dt is the acuumulation time
        # and 3DGS can be generated arbitrayly, thus it can be done!
        # what you need to do: make thses steps differentialable
        # RGB to LUV, all RGB to spectral to LUV is good, use RGB is good, for it can make it more easily to converge(more posible set)
        # but RGB is meaningless as input , only intensity graph can be generated

        #first try to change gt_image and image to LUV, get L and then change the loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss.backward()
        if args.event == True :
            # assert event_path==None, "No event file provided in event mode"
            # assert index == len(viewpoint_stack), "exceed error"    
            #before it use a pop func, thus that item is nolongger exist
            
            #i+1 - i
            index_next = index+1
            viewpoint_cam_next = viewpoint_stack[index_next]
            # viewpoint_cam_next = Interpolator.interpolate_pose_at_time((index+0.1)*Interpolator.dt)
            render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, bg)
            image_next = render_pkg_next["render"]
            img_diff = differentialable_event_simu(image,image_next,1)

            #i - (i-1),lego
            # index_pre = index-1
            # viewpoint_cam_pre = viewpoint_stack[index_pre]
            # # viewpoint_cam_next = Interpolator.interpolate_pose_at_time((index+0.1)*Interpolator.dt)
            # render_pkg_pre = render(viewpoint_cam_pre, gaussians, pipe, bg)
            # image_pre = render_pkg_pre["render"]
            
            # img_diff = differentialable_event_simu(image_pre,image,C=1)


            gt_image = viewpoint_Event_stack[index].original_image.cuda()
            gt_image = Normalize_event_frame(gt_image)

            # 将值归一化到[-1, 1]
            # img_diff = 2 * (img_diff - torch.min(img_diff)) / (torch.max(img_diff) -torch.min(img_diff))-1
            # gt_image = 2 * (gt_image -  torch.min(gt_image)) / (torch.max(gt_image) -  torch.min(gt_image))-1
            # opt.lambda_dssim = 0.9999999
            # Ll1 = opt.lambda_dssim * (1.0 - ssim(img_diff, gt_image))
            # loss =  Ll1
            Ll1 = l1_loss_gray(img_diff, gt_image)
            opt.lambda_dssim = 0
            loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_gray(img_diff, gt_image))

            #TODO pre may be better
            gt_image_intensity = viewpoint_cam.original_image.cuda()
            # gt_image_intensity = viewpoint_cam_pre.original_image.cuda()
            ##TODO hyper parameter
            # gt_image = gt_image*14
            Ll1 = l1_loss_gray(image, gt_image_intensity)
            loss2 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_gray(image, gt_image))
            Event_weight = 0.9
            loss = Event_weight*loss1 + (1-Event_weight)*loss2
            # torchvision.utils.save_image(gt_image, "gt_image.png")
            # torchvision.utils.save_image(img_diff, "img_diff.png")
            loss.backward()
            # Ll1 = 1.0 - ssim(img_diff, gt_image)
            # if 0 <= index <= 3:
            #     t1 =  random.randint(0, 6)
            #     t2 =  random.randint(0, 6)
            # elif len(viewpoint_stack) - 3 <= index <= len(viewpoint_stack) - 6:
            #     t1 =  random.randint(len(viewpoint_stack) - 9, len(viewpoint_stack) - 3)
            #     t2 =  random.randint(len(viewpoint_stack) - 9, len(viewpoint_stack) - 3)
            # else:
            #     t1 = random.randint(max(0, index - 3), min(len(viewpoint_stack) - 1, index + 3))
            #     t2 = random.randint(max(0, index - 3), min(len(viewpoint_stack) - 1, index + 3))
            # if t1>t2:
            #     temp = t2
            #     t2 = t1
            #     t1 = temp
            # view = Interpolator.interpolate_pose_at_time(t1*Interpolator.dt)
            # view_next = Interpolator.interpolate_pose_at_time(t2*Interpolator.dt)
            # img1 = render(view, gaussians, pipe, bg)["render"]
            # img2 = render(view_next, gaussians, pipe, bg)["render"]
            # img_diff_ran = differentialable_event_simu(img1,img2)
            # # with torch.no_grad:
            # #     gt_image_ran = events_data.display_events_accumu(ev_data,t1*Interpolator.dt,t2*Interpolator.dt)
            # #     gt_image_ran = Normalize_event_frame(gt_image_ran)
            # gt_image_ran = events_data.display_events_accumu(ev_data,t1*Interpolator.dt,t2*Interpolator.dt)
            # gt_image_ran = np.reshape(gt_image_ran, (3, gt_image_ran.shape[0],gt_image_ran.shape[1]))
            # gt_image_ran = gt_image_ran.astype(np.float32)  # Convert to float32 if necessary
            # gt_image_ran = torch.from_numpy(gt_image_ran)  # Co
            # gt_image_ran = Normalize_event_frame(gt_image_ran).to('cuda')
            # Ll2 = 1.0 - ssim(img_diff_ran, gt_image_ran)
            # loss =  Ll1 + Ll2
        elif args.gray == True:
            #TODO pre may be better
            gt_image = viewpoint_cam.original_image.cuda()
            # index_pre = index-1
            # viewpoint_cam_pre = viewpoint_stack[index_pre]
            # gt_image = viewpoint_cam_pre.original_image.cuda()
            ##TODO hyper parameter
            # gt_image = gt_image*14
            Ll1 = l1_loss_gray(image, gt_image)
            # torchvision.utils.save_image(gt_image, "gt_image.png")
            # torchvision.utils.save_image(image, "image.png")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_gray(image, gt_image))
            loss.backward()
        elif args.deblur == True:
            gt_image = viewpoint_cam.original_image.cuda()
            viewpoint_cam_blur = viewpoint_blurry_stack[index]
            gt_blur_image = viewpoint_cam_blur.original_image.cuda()
            ##TODO hyper parameter
            # gt_image = gt_image*14
            blur_alpha = 0.65
            Ll1 = (1-blur_alpha)*l1_loss_gray(image, gt_image) + blur_alpha*l1_loss(image,gt_blur_image)
            L_ssim = (1-blur_alpha)*(1.0 - ssim_gray(image, gt_image)) + blur_alpha*(1.0 - ssim(image, gt_blur_image))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * L_ssim
            loss.backward()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()


        iter_end.record()
        # no_grad, thus is not so important
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            #TODO， these judge how to change gaussian by status, how to fix it in Eventloss? 
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
