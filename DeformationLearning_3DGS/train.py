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
from torch import nn
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.util import get_expon_weight_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import matplotlib.pyplot as plt

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize = True).cuda()

# from utils.vgg_loss import VGGLoss
# vgg = VGGLoss().to("cuda") #default layer=8

# get current date and year
now = datetime.now()
date = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")
# print(date)
time = now.strftime("%H_%M")
# print("time:", time)
# path_to_tdImage=os.path.join(os.getcwd(), "output", "images", date+"_"+time[:2])
# path_to_tdMesh = os.path.join(os.getcwd(), "output", "meshes", date+"_"+time[:2])
# if not os.path.exists(path_to_tdImage):
#     os.mkdir(path_to_tdImage)
# if not os.path.exists(path_to_tdMesh):
#     os.mkdir(path_to_tdMesh)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

path_to_3WI = os.path.join(os.getcwd(), os.pardir, "3DSSL-WS23_IntuitiveAnimation")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.OBJ_helper import OBJ
from utils.Dataset_handler import Filehandler
from utils.pickel_io import dump_pckl

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, subject_id, expName, unique_str, ALLcam=False, NofFrame = 5, path_to_output = "", scale=1.0, configs = {"COG": 1e10, "CS": 1e10, "TV": 1.0}, flag_densification=False):
    # COG regularizer
    lambda_COG_regTerm = configs["COG"]
    # Scale regularize
    lambda_CS_regTerm = configs["CS"]
    lambda_tv = configs["TV"]
    subd = configs["subd"]
    render_numGauss = configs["render_numGauss"]

    print("sh degree:", dataset.sh_degree)
    l1_test = 0.0
    psnr_test = 0.0
    lpips_test = 0.0
    ssim_test = 0.0
    
    for id_frame in range(NofFrame): #0,.., NofFrame-1
        first_iter = 0
        
        memo = "cs:"+ str(lambda_CS_regTerm) + ",cog:"+ str(lambda_COG_regTerm)
        tb_writer, model_path = prepare_output_and_logger(dataset, unique_str, expName=expName, memo = memo, frame_counter = id_frame, path_to_output=path_to_output, subject_id=subject_id)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, expName = expName, ALLcam=ALLcam, frame_counter=id_frame, subject_id=subject_id, scale=scale, render_numGauss = render_numGauss, subd = subd)

        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        # bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        print(f"opacity is optimized: {gaussians._opacity.requires_grad}")
        print(f"average opacity: {gaussians._opacity.mean()}")

        initial_COG = torch.tensor(gaussians.get_nparray_xyz(), device="cuda")

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        print(f"max iter: {opt.iterations}")
        tv_weight_func = get_expon_weight_func(weight_init = lambda_tv, weight_final= lambda_tv*1e-4, max_steps=opt.iterations)
        for iteration in range(first_iter, opt.iterations + 1):
            tb_writer.add_scalar('weight_tv', tv_weight_func(iteration), iteration)
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            # print(bg)

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            # print(f"L1-loss: {Ll1}")
            ssim_loss = 1.0 - ssim(image, gt_image)
            
            # if iteration % 2 == 0:
            #     pred_vgg = image.unsqueeze(0)
            #     gt_vgg = gt_image.unsqueeze(0)

            #     if pred_vgg.max()>1.0:
            #         pred_vgg = pred_vgg / 255.0
            #     if gt_vgg.max()>1.0:
            #         gt_vgg = gt_vgg / 255.0

            #     pred_vgg = torch.clamp(pred_vgg, 0.0, 1.0)
            #     gt_vgg = torch.clamp(gt_vgg, 0.0, 1.0)

            #     vgg_loss = vgg(pred_vgg, gt_vgg)
            # else:
            #     vgg_loss = 0.0

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss)  # + 0.1 * vgg_loss

            if not flag_densification:
                # regularization term for the position of COG
                if lambda_COG_regTerm > 0.0:
                    current_COG = gaussians.get_xyz
                    cog_reg_term = ((((initial_COG - current_COG)**2).sum(dim=1))**2).mean()
                    loss += lambda_COG_regTerm * cog_reg_term
                if lambda_CS_regTerm > 0.0:
                    # regularization term for the covariance scale
                    covariance_scale = gaussians.get_scaling
                    # l1l2
                    cs_reg_term = (((covariance_scale**2).sum(dim=1))**2).mean()
                    loss+=lambda_CS_regTerm * cs_reg_term

            gaussians.optimizer.zero_grad(set_to_none = True)
            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                
                additional_losses = {"loss": loss}

                # Log and save
                im_save_path = os.path.join(model_path, "im_result")
                if (iteration in testing_iterations):
                    l1_test, lpips_test, ssim_test, psnr_test = training_report(tb_writer=tb_writer, iteration=iteration, Ll1=Ll1, loss=loss, l1_loss=l1_loss, elapsed=iter_start.elapsed_time(iter_end), additional_losses=additional_losses, testing_iterations=testing_iterations, scene=scene, renderFunc=render,
                                                                                renderArgs=(pipe, background), expName=expName, max_iter= opt.iterations, frame_counter=id_frame, im_save_path=im_save_path, subject_id=subject_id)
                
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, test=True)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    
                # Densification
                if flag_densification and render_numGauss == None:
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


                if iteration == opt.iterations-1:
                    l1_test, lpips_test, ssim_test, psnr_test = training_report(tb_writer=tb_writer, iteration=iteration, Ll1=Ll1, loss=loss, l1_loss=l1_loss, elapsed=iter_start.elapsed_time(iter_end), additional_losses=additional_losses, testing_iterations=testing_iterations, scene=scene, renderFunc=render,
                                                                                renderArgs=(pipe, background), expName=expName, max_iter= opt.iterations, frame_counter=id_frame, im_save_path=im_save_path, subject_id=subject_id)
                    scene.save(iteration, test=True)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # final evaluation
        l1_test, lpips_test, ssim_test, psnr_test = training_report(tb_writer=tb_writer, iteration=iteration, Ll1=Ll1, loss=loss, l1_loss=l1_loss, elapsed=iter_start.elapsed_time(iter_end), additional_losses=additional_losses, testing_iterations=testing_iterations, scene=scene, renderFunc=render,
                                                                    renderArgs=(pipe, background), expName=expName, max_iter= opt.iterations, frame_counter=id_frame, im_save_path=im_save_path, subject_id=subject_id)
        
    return l1_test, lpips_test, ssim_test, psnr_test, tb_writer

def prepare_output_and_logger(args, _unique_id=None, expName="", memo = "", frame_counter=0, instant=False, path_to_output="", subject_id=""):
    # if not args.model_path:
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str=os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #         if _unique_id != None:
    unique_str = _unique_id
    # args.model_path = os.path.join("./output/", unique_str, expName, str(frame_counter))
    if instant:
        args.model_path = os.path.join(os.getcwd(), "output", subject_id, _unique_id, "instant_neutralface_training")
    else:
        args.model_path = os.path.join(path_to_output, unique_str, expName, str(frame_counter))
    print("output path:", args.model_path)
        
    # Set up output folder
    # print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))) + "\n" + memo)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path

# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, expName, max_iter, frame_counter, im_save_path):
#     if tb_writer:
#         tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar('iter_time', elapsed, iteration)

#     l1_test, psnr_test = 0.0, 0.0
#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
#                 for idx, viewpoint in enumerate(config['cameras']):
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    
#                     # image = torch.clamp(gt_image + image, 0.0, 1.0)
#                     # print(image.shape)
#                     # print(gt_image.shape)

#                     if iteration == max_iter:
#                         plt.subplot(1, 2, 1)
#                         plt.imshow(image.detach().cpu().permute(1,2,0).numpy())
#                         plt.title(f"Rendered image at {iteration}")
#                         plt.subplot(1, 2, 2)
#                         plt.imshow(gt_image.detach().cpu().permute(1, 2, 0).numpy())
#                         plt.title(f"GT at {iteration}")
#                         plt.suptitle(f"Camera ID: {viewpoint.colmap_id}")
#                         path_to_plotImage = im_save_path
#                         if not os.path.exists(path_to_plotImage):
#                             os.mkdir(path_to_plotImage)
#                         if not os.path.exists(os.path.join(path_to_plotImage, str(frame_counter))):
#                             os.mkdir(os.path.join(path_to_plotImage, str(frame_counter)))
#                         print(viewpoint.colmap_id)
#                         print("output path: ", os.path.join(path_to_plotImage, str(frame_counter)))
#                         plt.savefig(os.path.join(path_to_plotImage, str(frame_counter), str(viewpoint.colmap_id)+ "_" + str(iteration) +'.png'))
                    
#                     # if tb_writer and (idx < 5):
#                     #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                     #     if iteration == testing_iterations[0]:
#                     #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

#                     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                     tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])      
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
#                 if tb_writer:
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
#         if tb_writer:
#             tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
#         torch.cuda.empty_cache()

#     return l1_test, psnr_test


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, additional_losses, testing_iterations, scene : Scene, renderFunc, renderArgs, expName, max_iter, frame_counter, im_save_path, subject_id):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if len(additional_losses.keys()) != 0:
            for key in additional_losses.keys():
                if key.endswith("attention"):
                    # print(additional_losses[key].shape)
                    print(additional_losses[key])
                    tb_writer.add_histogram('scene/'+str(key), additional_losses[key], iteration)
                else:
                    tb_writer.add_scalar('train_loss_patches/'+str(key), additional_losses[key], iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    l1_test = 0.0
    psnr_test = 0.0
    lpips_test = 0.0
    ssim_test = 0.0
    with torch.no_grad():
        if iteration <= max_iter-1:
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    
                        gt_image = viewpoint.original_image.to("cuda").clamp_max_(1.0)
                        image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"].clamp_max_(1.0)
                        # resize for interpretable metrics
                        # 3, H, W
                        resized_gt_image = gt_image[:, 302:1302, :]
                        resized_image = image[:, 302:1302, :]

                        print("gt_image: ", resized_gt_image.shape)
                        print("image: ", resized_image.shape)
                        
                        # print("gt_image min/max: ", torch.min(gt_image), torch.max(gt_image))
                        # print("image min/max: ", torch.min(image), torch.max(image))
                        # image = torch.clamp(gt_image + image, 0.0, 1.0)
                        # print(image.shape)
                        # print(gt_image.shape)

                        # if iteration == max_iter:
                            # plt.subplot(1, 2, 1)
                            # plt.imshow(image.detach().cpu().permute(1,2,0).numpy())
                            # plt.title(f"Rendered image at {iteration}")
                            # plt.subplot(1, 2, 2)
                            # plt.imshow(gt_image.detach().cpu().permute(1, 2, 0).numpy())
                            # plt.title(f"GT at {iteration}")
                            # plt.suptitle(f"Camera ID: {viewpoint.colmap_id}")
                            # path_to_plotImage = im_save_path
                            # if not os.path.exists(path_to_plotImage):
                            #     os.mkdir(path_to_plotImage)
                            # if not os.path.exists(os.path.join(path_to_plotImage, str(frame_counter))):
                            #     os.mkdir(os.path.join(path_to_plotImage, str(frame_counter)))
                            # print(viewpoint.colmap_id)
                            # print("output path: ", os.path.join(path_to_plotImage, str(frame_counter)))
                            # plt.savefig(os.path.join(path_to_plotImage, str(frame_counter), str(viewpoint.colmap_id)+ "_" + str(iteration) +'.png'))
                        # if tb_writer and (idx < 5):
                        #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        #     if iteration == testing_iterations[0]:
                        #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), resized_image[None], global_step=iteration)
                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), resized_gt_image[None], global_step=iteration)

                        _l1 = l1_loss(resized_image, resized_gt_image).mean().double()
                        _psnr = psnr(resized_image, resized_gt_image).mean().double()
                        _ssim = ssim(resized_image, resized_gt_image).mean().double()
                        _lpips = LPIPS(resized_image[None], resized_gt_image[None])

                        if config["name"]=='test':
                            l1_test += _l1
                            psnr_test += _psnr
                            ssim_test += _ssim
                            lpips_test += _lpips

                        if tb_writer:
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/l1".format(viewpoint.image_name), _l1, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/psnr".format(viewpoint.image_name), _psnr, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/lpips".format(viewpoint.image_name), _lpips, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/ssim".format(viewpoint.image_name), _ssim, iteration)
                
                        # shape should be: B, 3, h, w
                        
                        # print("LPIPS")
                        # print(gt_image.shape)
                        # print(resized_image.shape)

                    if config['name']=="test":
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        ssim_test /= len(config['cameras'])
                        lpips_test /= len(config['cameras'])

                    # if iteration == max_iter:
                    #     lpips_test /=len(config['cameras'])
                    #     # lpips_test = lpips_test.detach().cpu().numpy() + 0

                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                    if tb_writer:
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + f'/loss_viewpoint/{viewpoint.image_name} - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + f'/loss_viewpoint/{viewpoint.image_name} - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + f'/loss_viewpoint/{viewpoint.image_name} - lpips', lpips_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + f'/loss_viewpoint/{viewpoint.image_name} - ssim', ssim_test, iteration)
                        
            if tb_writer:
                # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()

    return l1_test, lpips_test, ssim_test, psnr_test

