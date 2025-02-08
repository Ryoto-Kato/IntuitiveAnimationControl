import torch
from scene import Scene
import os, sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import numpy as np
# training
from utils.loss_utils import l1_loss, ssim, l2_loss
from train import prepare_output_and_logger, training_report
from random import randint
from scene.dataset_readers import CameraInfo
from scene.cameras import Camera
"""
We need following opacities for gaussian rendering 
['xyz', 'f_dc', 'rotation', 'scale']
    -   xyz
    -   f_dc
    -   rotation
    -   scale
    -   opacity (infinity, 1.0)
    -   active_sh_degree = 0
"""

"python original_render.py --path_to_hdf5=./output/f336a291-bnotALLcam/3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5 --path_to_saveIMG=./output/f336a291-bnotALLcam/blendshape_result"

def instant_neutralFace_training(dataset, opt, pipe, gaussians:GaussianModel, scene:Scene, expName="E001_Neutral_Eyes_Open", unique_str="f336a291-bnotALLcam", ALLcam=False):
    # COG regularizer
    lambda_COG_regTerm = 1e-2
    # Scale regularizer
    lambda_CS_regTerm = 1e-2
    previous_gaussian = None
    opt.iterations = 500
    first_iter = 0
    gaussians = gaussians
    scene = scene
    
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    initial_COG = torch.tensor(gaussians.get_nparray_xyz(), device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # print(bg)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # print(f"L1-loss: {Ll1}")


        # regularization term for the position of COG
        current_COG = gaussians.get_xyz
        cog_reg_term = ((((initial_COG - current_COG)**2).sum(dim=1))**2).mean()
        # print(f"COG_reg_term: {cog_reg_term}")
        # print(f"initial COG: {initial_COG.sum()}")
        # print(f"gaussian.xyz: {current_COG.sum()}")

        # regularization term for the covariance scale
        covariance_scale = gaussians.get_scaling
        # l1l2
        cs_reg_term = (((covariance_scale**2).sum(dim=1))**2).mean()
        # print(f"CS_reg_term: {cs_reg_term}")
        
        # print(reg_term_l1l1)

        ssim_loss = 1.0 - ssim(image, gt_image)
        # print(f"SSIM_loss: {ssim_loss}")

        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss) + lambda_CS_regTerm * cs_reg_term + lambda_COG_regTerm * cog_reg_term
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss) + lambda_CS_regTerm * cs_reg_term
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss) + lambda_COG_regTerm * cog_reg_term

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
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


def render_blendshape(dataset: ModelParams, optimization: OptimizationParams, pipeline: PipelineParams, path_to_hdf5:str, path_to_saveIMG:str, sh_degree=3, coeffs = None, num_Blendshape_compos = 3, original_train_id = "f336a291-bnotALLcam", instant_train = True, dc_type = "pca"):
    # init gaussian
    save_path = path_to_saveIMG
    gaussians = GaussianModel(sh_degree=sh_degree)
    # init scene
    # render given 3D Gaussian in the same 38 camera configuratins
    # load camera configuration from "E001_Neutral_Eyes_Open" and first_time_stamp = "000102"
    scene = Scene(dataset, gaussians, render_blendshape = True, path_to_hdf5=path_to_hdf5, num_Blendshape_compos=num_Blendshape_compos, dc_type=dc_type)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    backgroud = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    # pre_traing with neutral face
    if instant_train:
        instant_neutralFace_training(dataset=dataset, opt=optimization, pipe = pipeline, gaussians=gaussians, scene=scene, unique_str=original_train_id)

    with torch.no_grad():
        if "xyz" in str(path_to_hdf5):
            scene.update_xyz_blendshape(coeffs=coeffs)
        else:
            scene.update_blendshape(coeffs=coeffs)
        # views = scene.getTrainCameras()
        views = scene.getTestCameras()
        for idx, view in enumerate(tqdm(views, desc = "Rendering progress")):
            if idx > 5:
                break
            camera_id = int(view.colmap_id)
            print(camera_id)
            rendered_img = render(view, gaussians, pipeline, backgroud)["render"]
            rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
            gt = view.original_image[0:3, :, :]
            gt = torch.clamp(gt, 0.0, 1.0)
            
            # plt.subplot(1, 2, 1)
            # plt.imshow(rendered_img.detach().cpu().permute(1, 2, 0).numpy())
            # plt.title(f"Rendered image")
            # plt.subplot(1, 2, 2)
            # plt.imshow(gt.detach().cpu().permute(1, 2, 0).numpy())
            # plt.title(f"GT")
            # plt.suptitle(f"Camera ID: {camera_id}")
            # path_to_plotImage = os.path.join(save_path, '{0:05d}'.format(camera_id) + ".png")
            # plt.savefig(path_to_plotImage)
            torchvision.utils.save_image(rendered_img, os.path.join(save_path, '{0:05d}'.format(camera_id) + "_render.png"))
            # torchvision.utils.save_image(gt, os.path.join(save_path, "gt.png"))

def render_blendshape_interpolation(dataset: ModelParams, optimization: OptimizationParams, pipeline: PipelineParams, path_to_hdf5:str, path_to_saveIMG:str, sh_degree=3, num_Blendshape_compos = 3, iterpolate_max= [3e-1, 3e-1, 3e-1], targets_blendshape_list = [1, 2, 3], step_size = 10, original_train_id = "f336a291-bnotALLcam", instant_train = True, dc_type = "pca", ):
    # init gaussian
    save_path = path_to_saveIMG
    num_Blendshape_compos = int(np.max(np.asarray(targets_blendshape_list)))

    gaussians = GaussianModel(sh_degree=sh_degree)
    # init scene
    # render given 3D Gaussian in the same 38 camera configuratins
    # load camera configuration from "E001_Neutral_Eyes_Open" and first_time_stamp = "000102"
    scene = Scene(dataset, gaussians, render_blendshape = True, path_to_hdf5=path_to_hdf5, num_Blendshape_compos=num_Blendshape_compos, dc_type=dc_type)
    # bg_color = [1,1,1]
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    backgroud = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # pre_traing with neutral face
    if instant_train:
        instant_neutralFace_training(dataset=dataset, opt=optimization, pipe = pipeline, gaussians=gaussians, scene=scene, unique_str=original_train_id)

    # one_step_size = np.asarray(iterpolate_max)
    # print(one_step_size)
    targets_blendshape_list = np.asarray(targets_blendshape_list, dtype=int)
    targets_blendshape_list = np.sort(targets_blendshape_list)
    print(targets_blendshape_list)
    
    for i, exp_idx in enumerate(targets_blendshape_list):
        exp_path_to_saveIMG = os.path.join(save_path,str(exp_idx)+"_comp")
        makedirs(path_to_saveIMG, exist_ok=True)
        coeffs = np.zeros(num_Blendshape_compos)

        for step_idx in range(step_size+1):
            # path_to_saveIMG = os.path.join(exp_path_to_saveIMG,str(step_idx))
            # makedirs(path_to_saveIMG, exist_ok=True)
            # interpolation of coefficients
            if step_idx == 0:
                coeffs[exp_idx-1] = 0.0
            else:
                coeffs[exp_idx-1] = (iterpolate_max[i])/step_size
            print(coeffs)
            print(f"{step_idx}/{step_size+1}")
            with torch.no_grad():
                if "xyz" in str(path_to_hdf5):
                    scene.update_xyz_blendshape(coeffs=coeffs)
                else:
                    scene.update_blendshape(coeffs=coeffs)
                
                scene.gaussians.save_ply(path=os.path.join(exp_path_to_saveIMG, str(exp_idx)+".ply"))
                # views = scene.getTrainCameras()
                views = scene.getTestCameras()
                # views = scene.getInterpCameras()
                for idx, view in enumerate(tqdm(views, desc = "Rendering progress")):
                    # if view.colmap_id == "400015" or view.colmap_id == "400048":
                    cam_path_to_saveIMG = os.path.join(exp_path_to_saveIMG, str(view.colmap_id))
                    makedirs(cam_path_to_saveIMG, exist_ok=True)
                    # if view.colmap_id == "400015" or view.colmap_id == "400048":
                    camera_id = int(view.colmap_id)
                    print(camera_id)
                    rendered_img = render(view, gaussians, pipeline, backgroud)["render"]
                    rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
                    # gt = view.original_image[0:3, :, :]
                    # gt = torch.clamp(gt, 0.0, 1.0)
                    
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(rendered_img.detach().cpu().permute(1, 2, 0).numpy())
                    # plt.title(f"Rendered image")
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(gt.detach().cpu().permute(1, 2, 0).numpy())
                    # plt.title(f"GT")
                    # plt.suptitle(f"Camera ID: {camera_id}")
                    # path_to_plotImage = os.path.join(path_to_saveIMG, '{0:05d}'.format(camera_id) + ".png")
                    # plt.savefig(path_to_plotImage)
                    torchvision.utils.save_image(rendered_img, os.path.join(cam_path_to_saveIMG, '{0:03d}'.format(step_idx) +"_render.png"))
                    # torchvision.utils.save_image(gt, os.path.join(path_to_saveIMG, '{0:05d}'.format(camera_id) + "_gt.png"))


def render_blendshape_cams_interpolation(dataset: ModelParams, optimization: OptimizationParams, pipeline: PipelineParams, path_to_hdf5:str, path_to_saveIMG:str, sh_degree=3, num_Blendshape_compos = 3, iterpolate_max= [3e-1, 3e-1, 3e-1], targets_blendshape_list = [1, 2, 3], step_size = 10, original_train_id = "f336a291-bnotALLcam", instant_train = True, dc_type = "pca", interp_views = None):
        # init gaussian
        save_path = path_to_saveIMG
        num_Blendshape_compos = int(np.max(np.asarray(targets_blendshape_list)))

        gaussians = GaussianModel(sh_degree=sh_degree)
        # init scene
        # render given 3D Gaussian in the same 38 camera configuratins
        # load camera configuration from "E001_Neutral_Eyes_Open" and first_time_stamp = "000102"
        scene = Scene(dataset, gaussians, render_blendshape = True, path_to_hdf5=path_to_hdf5, num_Blendshape_compos=num_Blendshape_compos, dc_type=dc_type, sp_interp_cams=interp_views)
        # bg_color = [1,1,1]
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        backgroud = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # pre_traing with neutral face
        if instant_train:
            instant_neutralFace_training(dataset=dataset, opt=optimization, pipe = pipeline, gaussians=gaussians, scene=scene, unique_str=original_train_id)

        # one_step_size = np.asarray(iterpolate_max)
        # print(one_step_size)
        targets_blendshape_list = np.asarray(targets_blendshape_list, dtype=int)
        targets_blendshape_list = np.sort(targets_blendshape_list)
        print(targets_blendshape_list)
        for i, exp_idx in enumerate(targets_blendshape_list):
            
            exp_path_to_saveIMG = os.path.join(save_path,str(exp_idx)+"_comp")
            makedirs(path_to_saveIMG, exist_ok=True)
            
            coeffs = np.zeros(num_Blendshape_compos)
            
            views = scene.getInterpCameras()
            num_steps_cam_interp = len(views)
            max_cam_steps_while_exp_interp = int(num_steps_cam_interp/step_size)
            start_view = 0
            end_view = max_cam_steps_while_exp_interp

            for step_idx in range(step_size):
                # path_to_saveIMG = os.path.join(exp_path_to_saveIMG,str(step_idx))
                # makedirs(path_to_saveIMG, exist_ok=True)
                # interpolation of coefficients
                coeffs[exp_idx-1] = (iterpolate_max[i])/step_size
                print(coeffs)
                print(f"{step_idx}/{step_size}")
                with torch.no_grad():
                    if "xyz" in str(path_to_hdf5):
                        scene.update_xyz_blendshape(coeffs=coeffs)
                    else:
                        scene.update_blendshape(coeffs=coeffs)
                    
                    scene.gaussians.save_ply(path=os.path.join(exp_path_to_saveIMG, str(exp_idx)+".ply"))
                    # views = scene.getTrainCameras()
                    # views = scene.getTestCameras()
                    for idx, view in enumerate(tqdm(views[start_view:end_view], desc = "Rendering progress")):
                        # if view.colmap_id == "400015" or view.colmap_id == "400048":
                        cam_path_to_saveIMG = os.path.join(exp_path_to_saveIMG)
                        # makedirs(cam_path_to_saveIMG, exist_ok=True)
                        # if view.colmap_id == "400015" or view.colmap_id == "400048":
                        camera_id = int(view.colmap_id)
                        print(camera_id)
                        rendered_img = render(view, gaussians, pipeline, backgroud)["render"]
                        rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
                        # gt = view.original_image[0:3, :, :]
                        # gt = torch.clamp(gt, 0.0, 1.0)
                        
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(rendered_img.detach().cpu().permute(1, 2, 0).numpy())
                        # plt.title(f"Rendered image")
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(gt.detach().cpu().permute(1, 2, 0).numpy())
                        # plt.title(f"GT")
                        # plt.suptitle(f"Camera ID: {camera_id}")
                        # path_to_plotImage = os.path.join(path_to_saveIMG, '{0:05d}'.format(camera_id) + ".png")
                        # plt.savefig(path_to_plotImage)
                        torchvision.utils.save_image(rendered_img, os.path.join(cam_path_to_saveIMG, '{0:05d}'.format(view.colmap_id) +"_render.png"))
                        # torchvision.utils.save_image(gt, os.path.join(path_to_saveIMG, '{0:05d}'.format(camera_id) + "_gt.png"))
                temp_end_view = end_view
                end_view = end_view + max_cam_steps_while_exp_interp
                start_view = temp_end_view
        return views
                

"""
    conda activate pytransform3d_3dgs
    python original_render.py --path_to_hdf5="./output/f336a291-bnotALLcam/gauss_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs_0.85_lambda2.0.hdf5" --path_to_saveIMG=./output/f336a291-bnotALLcam/blendshape_result --dc_type=sldc
"""

if __name__ == "__main__":


    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering Parameters")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--path_to_hdf5", type = str, default = None)
    parser.add_argument("--path_to_saveIMG", type = str, default=None)
    parser.add_argument("--num_blendshape_compos", type = int, default=3)
    parser.add_argument("--dc_type", type = str, default="pca")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--coeffInterp", action="store_true", default=False)
    parser.add_argument("--camInterp", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.random_background = True
    args.sh_degree = 3
    path_to_saveIMG = args.path_to_saveIMG
    safe_state(args.quiet)

    # [TODO] set the training session ID
    original_train_id = ""
    
    _required_instant_train = True
    print("Dc type:", args.dc_type)

    # render image without interpolation
    if not args.coeffInterp:
        for i in range(0, 10):
            args.path_to_saveIMG = os.path.join(path_to_saveIMG,str(i+1)+"_comp")
            makedirs(args.path_to_saveIMG, exist_ok=True)
            coeffs = np.zeros(i+1)
            if args.dc_type == "pca":
                coeffs[i] = 2e2 # xyz = 5.0, All = 2e2
            elif args.dc_type == "sldc":
                coeffs[i] = 5e-1
            render_blendshape(model.extract(args), optimization.extract(args), pipeline.extract(args), path_to_hdf5=args.path_to_hdf5, path_to_saveIMG = args.path_to_saveIMG, sh_degree=args.sh_degree, coeffs = coeffs, num_Blendshape_compos = i+1, original_train_id=original_train_id, instant_train=_required_instant_train, dc_type=args.dc_type)

    else:
        # render image with interpolation (from neutral to target facial expression)
        Nblendshapes = 20
        step_size = 20
        list_iteration_max = np.zeros(1)
        target_exps = np.zeros(1)
        interp_views = None
        for i in range(0, 10):
            print(f"Render {i}-th component of {args.dc_type}")
            if args.dc_type == "pca":
                list_iteration_max[0] = 2e2
            elif args.dc_type == "sldc":
                list_iteration_max[0] = 5e-1
            target_exps[0] = int(i+1)
            
            if args.camInterp:
                interp_views = render_blendshape_cams_interpolation(model.extract(args), optimization.extract(args), pipeline.extract(args), path_to_hdf5=args.path_to_hdf5, path_to_saveIMG = args.path_to_saveIMG, sh_degree=args.sh_degree, num_Blendshape_compos = Nblendshapes, iterpolate_max= list_iteration_max, targets_blendshape_list = target_exps, step_size = step_size, original_train_id=original_train_id, instant_train=_required_instant_train, dc_type=args.dc_type, interp_views=interp_views)
            else:
                render_blendshape_interpolation(model.extract(args), optimization.extract(args), pipeline.extract(args), path_to_hdf5=args.path_to_hdf5, path_to_saveIMG = args.path_to_saveIMG, sh_degree=args.sh_degree, num_Blendshape_compos = Nblendshapes, iterpolate_max= list_iteration_max, targets_blendshape_list = target_exps, step_size = step_size, original_train_id=original_train_id, instant_train=_required_instant_train, dc_type=args.dc_type)
            

        
