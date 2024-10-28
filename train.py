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
from utils.loss_utils import *
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr,LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from Event_sensor.event_tools import *
# import copy
# from Event_sensor.src.event_buffer import EventBuffer
# import torchvision
# from render import Generate_new_view
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import random
import torch.optim as optim
def generate_random_integer_nearby(target_integer, range_half_width):
    """
    Generate a random integer nearby the target_integer within the specified range_half_width.
    """
    lower_bound = target_integer - range_half_width
    upper_bound = target_integer + range_half_width
    return random.randint(lower_bound, upper_bound)

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
    # bg_color = [0.651,0.651,0.651]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    c = torch.nn.Parameter(torch.tensor(0.17))
    optimizer_c = optim.Adam([c], lr=0.1)  # 设置 c 的学习率为 0.001
    optimizer_c.zero_grad()
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
            # if args.ColorEvent ==True:
            #     viewpoint_Event_R_stack = scene.getEventCameras_R().copy()
            #     viewpoint_Event_G_stack = scene.getEventCameras_G().copy()
            #     viewpoint_Event_B_stack = scene.getEventCameras_B().copy()


        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if args.event == True:
            #index = randint(0, len(viewpoint_stack)-2)
            index = randint(2, len(viewpoint_stack)-4)
            opt.opacity_reset_interval = 10000
            # if index == 48 or index == 49:
            #     index=3
        else:
            # index = randint(0, len(viewpoint_stack)-1)
            index = randint(2, len(viewpoint_stack)-3)

        #colmap do not support additional test images
        #thus we use manully test
        #test on 5,25,45,65,85
        if args.event == True or args.gray == True:
            if index == 5 or index == 25 or index == 45 or index == 65 or index == 85:
                index=index -1
        
        # viewpoint_cam = viewpoint_stack.pop(index)
        #delete some bad img
            
        # index = 2
        viewpoint_cam = viewpoint_stack[index]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # torchvision.utils.save_image(image, "test.png")


        if args.event == True :
            viewpoint_stack_r = viewpoint_Event_stack
            index_now = index
            index_next = index+1
            viewpoint_cam_now = viewpoint_stack_r[index_now]
            viewpoint_cam_next = viewpoint_stack_r[index_next]

            # render_pkg_now = render(viewpoint_cam_now, gaussians, pipe, bg)
            
            # image_now = render_pkg_now["render"]
            render_pkg_now = render(viewpoint_cam_now, gaussians, pipe, bg)
            image_now = render_pkg_now["render"]
            render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, bg)
            image_next = render_pkg_next["render"]
                        

            img_diff = differentialable_event_simu(image_now,image_next,False,c)

            #upper bound
            image_now_gt = viewpoint_cam_now.original_image.cuda()
            image_next_gt = viewpoint_cam_next.original_image.cuda()
            gt_image = differentialable_event_simu(image_now_gt,image_next_gt,False,0.17)


            # Ll1 = l1_loss_gray_event(img_diff, gt_image)
            Ll1 = l1_loss(img_diff, gt_image)
            # Ll1 = l1_loss_gray(img_diff, gt_image)

            opt.lambda_dssim = 0
            loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_gray(img_diff, gt_image))

            gt_image_intensity = viewpoint_cam.original_image.cuda()
            # gt_image_intensity = viewpoint_cam_pre.original_image.cuda()
            # gt_image = gt_image * 14
            # Ll1 = l1_loss_gray(image, gt_image_intensity)
            Ll1 = l1_loss(image, gt_image_intensity)

            loss2 = (1.0 - opt.lambda_dssim) * Ll1
            Event_weight = 0.9

            # 创建掩码
            mask = (gt_image != 0).float()

            # 计算最终的loss
            loss = Event_weight * (loss1 * mask).sum() + (1 - Event_weight) * (loss2 * (1 - mask)).sum()

            # 平均化 loss
            loss /= (mask.sum() + (1 - mask).sum())
            if args.deblur == True:
                viewpoint_cam_blur = viewpoint_blurry_stack[index]            
                gt_blur_image = viewpoint_cam_blur.original_image.cuda()
                # gt_image = gt_image*14
                blur_alpha = 0.5
                Ll1 = l1_loss(image,gt_blur_image)
                loss = (1.0 - blur_alpha) * loss + blur_alpha *  Ll1
            # torchvision.utils.save_image(image_now_gt, "image_now.png")
            # torchvision.utils.save_image(image_next_gt, "image_next.png")
            # torchvision.utils.save_image(image_now[0,:,:], "image_now_3DGS.png")
            # torchvision.utils.save_image(image_next[0,:,:], "image_next_3DGS.png")
            # torchvision.utils.save_image(gt_image, "gt_image.png")
            # torchvision.utils.save_image(img_diff, "img_diff.png")
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_c.step()
        elif args.gray == True:
            gt_image = viewpoint_cam.original_image.cuda()
            # index_pre = index-1
            # viewpoint_cam_pre = viewpoint_stack[index_pre]
            # gt_image = viewpoint_cam_pre.original_image.cuda()
            # gt_image = gt_image*14
            Ll1 = l1_loss_gray(image, gt_image)
            # torchvision.utils.save_image(gt_image, "gt_image.png")
            # torchvision.utils.save_image(image, "image.png")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_gray(image, gt_image))
            loss.backward()
        # elif args.ColorEvent ==True:
        #     viewpoint_stack_R = viewpoint_Event_R_stack
        #     viewpoint_stack_G = viewpoint_Event_G_stack
        #     viewpoint_stack_B = viewpoint_Event_B_stack

        #     index_now = index
        #     index_next = index+1
        #     viewpoint_cam_now = viewpoint_stack_R[index_now]
        #     viewpoint_cam_next = viewpoint_stack_R[index_next]

        #     # render_pkg_now = render(viewpoint_cam_now, gaussians, pipe, bg)
            
        #     # image_now = render_pkg_now["render"]
        #     render_pkg_now = render(viewpoint_cam_now, gaussians, pipe, bg)
        #     image_now = render_pkg_now["render"]
        #     render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, bg)
        #     image_next = render_pkg_next["render"]
                        

        #     # img_diff_R = differentialable_event_simu(image_now[0,:,:],image_next[0,:,:],True,c)
        #     # img_diff_G = differentialable_event_simu(image_now[1,:,:],image_next[1,:,:],True,c)
        #     # img_diff_B = differentialable_event_simu(image_now[2,:,:],image_next[2,:,:],True,c)

        #     img_diff_B = differentialable_event_simu(image_now[0,:,:],image_next[0,:,:],True,c)
        #     img_diff_G = differentialable_event_simu(image_now[1,:,:],image_next[1,:,:],True,c)
        #     img_diff_R = differentialable_event_simu(image_now[2,:,:],image_next[2,:,:],True,c)

        #     # img_diff_G = differentialable_event_simu(image_now[0,:,:],image_next[0,:,:],True,c)
        #     # img_diff_B = differentialable_event_simu(image_now[1,:,:],image_next[1,:,:],True,c)
        #     # img_diff_R = differentialable_event_simu(image_now[2,:,:],image_next[2,:,:],True,c)

        #     #upper bound
        #     image_now_gt_R = viewpoint_stack_R[index_now].original_image.cuda()
        #     image_next_gt_R = viewpoint_stack_R[index_next].original_image.cuda()
        #     gt_image_R = differentialable_event_simu(image_now_gt_R,image_next_gt_R,False,0.17)
        #     image_now_gt_G = viewpoint_stack_G[index_now].original_image.cuda()
        #     image_next_gt_G = viewpoint_stack_G[index_next].original_image.cuda()
        #     gt_image_G = differentialable_event_simu(image_now_gt_R,image_next_gt_R,False,0.17)
        #     image_now_gt_B = viewpoint_stack_B[index_now].original_image.cuda()
        #     image_next_gt_B = viewpoint_stack_B[index_next].original_image.cuda()
        #     gt_image_B = differentialable_event_simu(image_now_gt_R,image_next_gt_R,False,0.17)

        #     Ll1 = (0.299*l1_loss(img_diff_R, gt_image_R)+0.587*l1_loss(img_diff_G, gt_image_G)+0.114*l1_loss(img_diff_B, gt_image_B))

        #     opt.lambda_dssim = 0
        #     loss1 = (1.0 - opt.lambda_dssim) * Ll1

        #     gt_image_intensity = viewpoint_cam.original_image.cuda()
        #     # gt_image_intensity = viewpoint_cam_pre.original_image.cuda()
        #     # gt_image = gt_image*14
        #     Ll1 = l1_loss_gray(image, gt_image_intensity)
        #     loss2 = (1.0 - opt.lambda_dssim) * Ll1
        #     Event_weight = 0.99
        #     loss = Event_weight*loss1 + (1-Event_weight)*loss2
        #     if args.deblur == True:
        #         viewpoint_cam_blur = viewpoint_blurry_stack[index]            
        #         gt_blur_image = viewpoint_cam_blur.original_image.cuda()
        #         # gt_image = gt_image*14
        #         blur_alpha = 0.5
        #         Ll1 = l1_loss(image,gt_blur_image)
        #         loss = (1.0 - blur_alpha) * loss + blur_alpha *  Ll1
        #     # torchvision.utils.save_image(image_now_gt, "image_now.png")
        #     # torchvision.utils.save_image(image_next_gt, "image_next.png")
        #     # torchvision.utils.save_image(gt_image, "gt_image.png")
        #     # torchvision.utils.save_image(img_diff, "img_diff.png")
        #     optimizer_c.zero_grad()
        #     loss.backward()
        #     optimizer_c.step()
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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[399,999,1399,1699,1999,2999,3999,4999,5999,6999,7999,8999,9999,10999,13999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[399,999,1399,1699,1999,2999,4999,3999,5999,6999,7999,8999,9999,10999,13999])
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
