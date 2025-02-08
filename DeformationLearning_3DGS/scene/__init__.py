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
import sys
import random
import json
import h5py
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BlendshapeProp, GaussianBlenshapeProp
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from nersemble_dataset_readers import readNersembleSceneInfo
from multiface_dataset_readers import readMultifaceSceneInfo
from utils.quaterion_slerp import transform_interpolation
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.general_utils import PILtoTorch

path_to_3WI = os.path.join(os.getcwd(), os.pardir, "3DSSL-WS23_IntuitiveAnimation", "")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.pickel_io import dump_pckl, load_from_memory

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], expName = "E001_Neutral_Eyes_Open", ALLcam = False, frame_counter = 0, render_blendshape = False, path_to_hdf5 = None, num_Blendshape_compos = 3, dc_type = "pca", sp_interp_cams=None, subject_id=0, scale = 1.0, subd=2, render_numGauss = None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.neutral_gaussians = None

        self.num_Blendshape_compos = num_Blendshape_compos
        self.bp = None
        print(path_to_hdf5)
        self.blendshape_type = "ALL"

        self.dc_type = dc_type

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(args.source_path)
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # scene_info = readMultifaceSceneInfo(expName=expName, frame_counter=frame_counter, ALLcam=ALLcam)
            scene_info = readNersembleSceneInfo(expName=expName, frame_counter=frame_counter, subject_id=subject_id, ALLcam=ALLcam, scale = scale, subd=subd, render_numGauss=render_numGauss)
        self.scene_info = scene_info
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        

        if render_blendshape:
            print("render blendshape")
            self.interps_cameras = {}
            if sp_interp_cams !=None:
                self.interps_cameras[resolution_scale] = sp_interp_cams
            else:
                target_cameras = self.getTrainCameras()
                target_camera_infos = scene_info.train_cameras

                num_steps = 20
                T = np.linspace(0, 1, num_steps)
                interp_camera_ids = [400004, 400013, 400016, 400060, 400063, 400069, 400049, 400026]
                # interp_camera_ids = [400013, 400016]
                # interp_camera_ids = sorted(interp_camera_ids, reverse=True)
                
                selected_target_cams = []
                selected_target_cam_infos = []
                for id, (target_camera, target_camera_info) in enumerate(zip(target_cameras, target_camera_infos)):
                    if int(target_camera.colmap_id) in interp_camera_ids:
                        selected_target_cams.append(target_camera)
                        selected_target_cam_infos.append(target_camera_info)
                
                selected_target_cams_infos = sorted(selected_target_cam_infos.copy(), key=lambda x: x.uid, reverse=True)
                selected_target_cams = sorted(selected_target_cams.copy(), key = lambda x: x.colmap_id, reverse=True)
                
                interp_common_info = selected_target_cam_infos[0]

                interp_camera_id = 0
                interp_cam_infos = []

                for i, (camera, scene_cam_info) in enumerate(zip(selected_target_cams, selected_target_cam_infos)):
                    # print("cameara id:", camera.colmap_id)
                    # print("scene_cam_id:", scene_cam_info.uid)
                    if i < len(selected_target_cams)-1:
                        next_cam = selected_target_cams[i+1]
                    else:
                        next_cam = selected_target_cams[0]
                    # print("next_cam id:", next_cam.colmap_id)
                    start_R = camera.R
                    start_T = camera.T
                    end_R = next_cam.R
                    end_T = next_cam.T

                    for id_step, t in enumerate(T):
                        interp_camera_id = interp_camera_id+1
                        # print(interp_camera_id)
                        interp_R, interp_T = transform_interpolation(start_R=start_R, end_R=end_R, start_t=start_T, end_t=end_T, time_step=t)
                        interp_cam_info = CameraInfo(uid = interp_camera_id, R = interp_R, T = interp_T, K = interp_common_info.K, FovX=interp_common_info.FovX, FovY=interp_common_info.FovY, image = interp_common_info.image, image_path=interp_common_info.image_path, image_name=interp_common_info.image_name, width=interp_common_info.width, height=interp_common_info.height)
                        interp_cam_infos.append(interp_cam_info)
                
                self.interps_cameras[resolution_scale] = cameraList_from_camInfos(interp_cam_infos, resolution_scale, args)
                # print("number of interpolated cameras: ", len(self.interps_cameras))
                    

            if "xyz" in str(path_to_hdf5):
                self.blendshape_type = "xyz"
            else:
                self.blendshape_type = "ALL"


            print(f"{dc_type} and {self.blendshape_type}")
            if self.blendshape_type == "ALL":
                print(f"-------{self.blendshape_type}-------")
                f = h5py.File(path_to_hdf5, 'r')
                MEAN = np.asarray(f['MEAN'])
                STD = np.asarray(f['STD'])
                # STD = STD.reshape(STD.shape[0], 1)
                print("Number of attributes: ", MEAN.shape[0])
                dset = f['dim_info']
                list_attrib = str(dset.attrs['list_attrName'])
                dim_info = np.asarray(f['dim_info'])
                tris= np.asarray(f['tris'])
                if dc_type == "sldc":
                    pca_C = pca_W = mbspca_C = mbspca_W = None
                    sldc_C = np.asarray(f['sldc_C'])
                    sldc_W = np.asarray(f['sldc_W'])
                else:
                    pca_C= np.asarray(f['pca_C'])
                    pca_W= np.asarray(f['pca_W'])
                    mbspca_C= np.asarray(f['mbspca_C'])
                    mbspca_W= np.asarray(f['mbspca_W'])
                    sldc_C = None
                    sldc_W = None

                print("PCA_C", pca_C.shape)
                print("STD", STD.shape)

                self.bp = BlendshapeProp(MEAN=MEAN, STD = STD, list_attrib=list_attrib, dim_info=dim_info, tris = tris, pca_C=pca_C, pca_W=pca_W, mbspca_C= mbspca_C, mbspca_W=mbspca_W, sldc_C=sldc_C, sldc_W=sldc_W)
                coeffs = np.zeros(self.num_Blendshape_compos)
                # result of blendshaping
                gbp = self.gaussians.blenshape_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type, average=True)
                self.gaussians.creat_from_GaussianBlendshapeProp(gbp=gbp)
                self.neutral_gaussians = self.gaussians
                # self.gaussians.save_ply(path = os.path.join(os.getcwd() , "samples", "GBP.ply"))
            elif self.blendshape_type == "xyz":
                print(f"-------{self.blendshape_type}-------")
                f = h5py.File(path_to_hdf5, 'r')

                print(list(f.keys()))

                MEAN = np.asarray(f['MEAN'])
                STD = np.asarray(f["STD"])
                
                list_attrib = ['xyz', 'f_dc', 'rotation', 'scale']
                if "87652" in path_to_hdf5:
                    dim_info = np.asarray([262956, 262956, 350608, 262956])
                elif "5509" in path_to_hdf5:
                    dim_info = np.asarray([16527, 16527, 22036, 16527])
                
                if dc_type == "sldc":
                    pca_C = pca_W = mbspca_C = mbspca_W = None
                    sldc_C = np.asarray(f['sldc_C'])
                    sldc_C = sldc_C.reshape(sldc_C.shape[0], -1)
                    sldc_W = np.asarray(f['sldc_W'])
                    sldc_W = sldc_W.reshape(sldc_W.shape[0], -1)
                    print(f"SLDC_C shape: {sldc_C.shape}")
                    print(f"SLDC_W shape: {sldc_W.shape}")
                else:
                    pca_C= np.asarray(f['pca_C'])
                    pca_W= np.asarray(f['pca_W'])
                    mbspca_C= np.asarray(f['mbspca_C'])
                    mbspca_W= np.asarray(f['mbspca_W'])
                    sldc_C = None
                    sldc_W = None
                tris = np.asarray(f['tris'])

                self.bp = BlendshapeProp(MEAN=MEAN, STD = STD, list_attrib=list_attrib, dim_info=dim_info, tris = tris, pca_C=pca_C, pca_W=pca_W, mbspca_C= mbspca_C, mbspca_W=mbspca_W, sldc_C=sldc_C, sldc_W=sldc_W)
                coeffs = np.zeros(self.num_Blendshape_compos)
                # result of blendshaping
                gbp = self.gaussians.blenshape_xyz_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type, average=True)
                self.gaussians.creat_from_GaussianBlendshapeProp(gbp=gbp)
        else:
            print("training")
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    def get_gaussians(self):
        return self.gaussians
    
    def update_blendshape(self, coeffs):
        assert coeffs.shape[0] == self.num_Blendshape_compos
        gbp = self.gaussians.blenshape_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type)
        self.gaussians.update_from_GaussianBlendshapeProp(gbp=gbp)
    
    def update_xyz_blendshape(self, coeffs):
        assert coeffs.shape[0] == self.num_Blendshape_compos
        # neutral_gbp = self.gaussians.blenshape_reset(bp = self.bp)
        # self.gaussians.creat_from_GaussianBlendshapeProp(gbp=neutral_gbp)
        gbp = self.gaussians.blenshape_xyz_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type)
        self.gaussians.update_from_GaussianBlendshapeProp(gbp=gbp)
    
    def save(self, iteration, test=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_GaussianProp(os.path.join(point_cloud_path))
        if test:
            print("Load pickel")
            gaussian_prop=load_from_memory(path_to_memory=point_cloud_path, pickle_fname="gaussian_prop.pkl")
            print(gaussian_prop)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getInterpCameras(self, scale=1.0):
        return self.interps_cameras[scale]