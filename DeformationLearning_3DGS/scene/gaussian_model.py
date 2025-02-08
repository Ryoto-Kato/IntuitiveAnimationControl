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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import sys
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from dataclasses import dataclass
import h5py

path_to_3WI = os.path.join(os.getcwd(), os.pardir)
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.pickel_io import dump_pckl, load_from_memory

@dataclass
class GaussianProp:
        """
        f_dc is SH_coeffs not RGB!!
        """
        xyz: np.ndarray
        normals: np.ndarray
        f_dc: np.ndarray #this is SH_coeffs, needs to be converted to RGB by SH2RGB
        f_rest: np.ndarray
        opacities: np.ndarray
        scale: np.ndarray
        rotation: np.ndarray
        covariance: np.ndarray

@dataclass
class GaussianBlenshapeProp:
    # ['xyz', 'f_dc', 'rotation', 'scale']
    xyz: np.ndarray
    f_dc: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    opacities: np.ndarray
    active_sh_degree: int

@dataclass
class BlendshapeProp:
    # ['MEAN', 'STD', 'dataMat', 'dim_info', 'faceMask', 'mbspca_C', 'mbspca_W', 'pca_C', 'pca_W', 'tris']
    MEAN: np.ndarray
    STD: np.ndarray
    list_attrib: str
    dim_info: np.ndarray
    tris: np.ndarray
    pca_C: np.ndarray
    pca_W: np.ndarray
    mbspca_C: np.ndarray
    mbspca_W: np.ndarray
    sldc_C: np.ndarray
    sldc_W: np.ndarray

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.parameters = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    def blenshape_reset(self, bp:BlendshapeProp):
        deformed_gaussian = bp.MEAN # all attributes
        trained_f_dc = self._features_dc.detach().cpu().numpy()
        trained_f_dc = trained_f_dc.flatten().ravel()
        start_fdc = bp.dim_info[0]
        end_fdc = bp.dim_info[0] + bp.dim_info[1]
        deformed_gaussian[start_fdc:end_fdc]=trained_f_dc
        # compose GaussianBlenshapeProp
        # verts 5509
        # [xyz, f_dc, rotation, scale]
        # [16527, 16527, 22036, 16527]
        # verts 87652
        # [xyz, f_dc, rotation, scale]
        # [262956, 262956, 350608, 262956]
            
        start_column = 0
        end_column = 0
        list_attrib = ["xyz", "f_dc", "rotation", "scale"]
        attribues = {}
        numGauss = int(bp.dim_info[0]/3)
        for i, dim in enumerate(bp.dim_info):
            end_column += dim
            attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
            start_column +=dim
        opacities = np.full((numGauss, 1), np.inf)
        gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=3)
        return gbp
    
    def blenshape_xyz_computation(self, bp:BlendshapeProp, coeffs:np.ndarray, Ncomps= 10, dc_type = "pca", average = False):
        assert coeffs.shape[0] == Ncomps
        deformed_gaussian = bp.MEAN # all attributes
        print(deformed_gaussian.shape)
        Ncolumns_xyz = bp.dim_info[0]
        if average == False:
            trained_f_dc = self._features_dc.detach().cpu().numpy()
            trained_f_dc = trained_f_dc.flatten().ravel()
            start_fdc = bp.dim_info[0]
            end_fdc = bp.dim_info[0] + bp.dim_info[1]
            deformed_gaussian[start_fdc:end_fdc]=trained_f_dc
        
        if dc_type == "pca":
            print(f"-----{dc_type}-----")
            # deformed only xyz
            denormalized_xyz_dc = bp.pca_C[:Ncomps, :]*bp.STD[None, :Ncolumns_xyz]
            for i, coeff in enumerate(coeffs):
                for j, element in enumerate(denormalized_xyz_dc[i, :]):
                    deformed_gaussian[j] += (coeff * element)
            
            # compose GaussianBlenshapeProp
            # verts 5509
            # [xyz, f_dc, rotation, scale]
            # [16527, 16527, 22036, 16527]
            # verts 87652
            # [xyz, f_dc, rotation, scale]
            # [262956, 262956, 350608, 262956]
            
            start_column = 0
            end_column = 0
            list_attrib = ["xyz", "f_dc", "rotation", "scale"]
            attribues = {}
            numGauss = int(bp.dim_info[0]/3)
            for i, dim in enumerate(bp.dim_info):
                end_column += dim
                attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
                start_column +=dim
            opacities = np.full((numGauss, 1), np.inf)
            gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=0)
            return gbp
        elif dc_type == "mbspca":
            print(f"-----{dc_type}-----")
            denormalized_xyz_dc = bp.mbspca_C[:Ncomps, :]*bp.STD[None, :Ncolumns_xyz]
            for i, coeff in enumerate(coeffs):
                for j, element in enumerate(denormalized_xyz_dc[i, :]):
                    deformed_gaussian[j] += (coeff * element)
            
            # compose GaussianBlenshapeProp
            # verts 5509
            # [xyz, f_dc, rotation, scale]
            # [16527, 16527, 22036, 16527]
            # verts 87652
            # [xyz, f_dc, rotation, scale]
            # [262956, 262956, 350608, 262956]
            
            start_column = 0
            end_column = 0
            list_attrib = ["xyz", "f_dc", "rotation", "scale"]
            attribues = {}
            numGauss = int(bp.dim_info[0]/3)
            for i, dim in enumerate(bp.dim_info):
                print(attribues.keys())
                end_column += dim
                attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
                start_column +=dim
            opacities = np.full((numGauss, 1), np.inf)
            gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=3)
            return gbp
        elif dc_type == "sldc":
            print(f"-----{dc_type}-----")
            denormalized_xyz_dc = bp.sldc_C[:Ncomps, :]*bp.STD[None, :Ncolumns_xyz]
            for i, coeff in enumerate(coeffs):
                for j, element in enumerate(denormalized_xyz_dc[i, :]):
                    deformed_gaussian[j] += (coeff * element)
            
            # compose GaussianBlenshapeProp
            # verts 5509
            # [xyz, f_dc, rotation, scale]
            # [16527, 16527, 22036, 16527]
            # verts 87652
            # [xyz, f_dc, rotation, scale]
            # [262956, 262956, 350608, 262956]
            
            start_column = 0
            end_column = 0
            list_attrib = ["xyz", "f_dc", "rotation", "scale"]
            attribues = {}
            numGauss = int(bp.dim_info[0]/3)
            for i, dim in enumerate(bp.dim_info):
                end_column += dim
                attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
                start_column +=dim
            opacities = np.full((numGauss, 1), np.inf)
            gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=3)
            return gbp

    def blenshape_computation(self, bp:BlendshapeProp, coeffs:np.ndarray, Ncomps= 10, dc_type = "pca", average = False):
        assert coeffs.shape[0] == Ncomps
        deformed_gaussian = bp.MEAN
        if average == False:
            trained_f_dc = self._features_dc.detach().cpu().numpy()
            trained_f_dc = trained_f_dc.flatten().ravel()
            start_fdc = bp.dim_info[0]
            end_fdc = bp.dim_info[0] + bp.dim_info[1]
            deformed_gaussian[start_fdc:end_fdc]=trained_f_dc

        if dc_type == "pca":
            
            denormalized_dc = bp.pca_C[:Ncomps, :]*bp.STD[None, :]
            for i, coeff in enumerate(coeffs):
                for j, element in enumerate(denormalized_dc[i, :]):
                    deformed_gaussian[j] += (coeff * element)
            
            # compose GaussianBlenshapeProp
            # verts 5509
            # [xyz, f_dc, rotation, scale]
            # [16527, 16527, 22036, 16527]
            # verts 87652
            # [xyz, f_dc, rotation, scale]
            # [262956, 262956, 350608, 262956]
            
            start_column = 0
            end_column = 0
            list_attrib = ["xyz", "f_dc", "rotation", "scale"]
            attribues = {}
            numGauss = int(bp.dim_info[0]/3)
            for i, dim in enumerate(bp.dim_info):
                end_column += dim
                attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
                start_column +=dim
            opacities = np.full((numGauss, 1), np.inf)
            gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=3)
            return gbp
        elif dc_type == "mbspca":
            print(dc_type)
            denormalized_dc = bp.mbspca_C[:Ncomps, :]*bp.STD[None, :]
            for i, coeff in enumerate(coeffs):
                for j, element in enumerate(denormalized_dc[i, :]):
                    if abs(element) < 1e-8:
                        deformed_gaussian[j] += 0
                    else:
                        deformed_gaussian[j] += (coeff * element)
            
            # compose GaussianBlenshapeProp
            # verts 5509
            # [xyz, f_dc, rotation, scale]
            # [16527, 16527, 22036, 16527]
            # verts 87652
            # [xyz, f_dc, rotation, scale]
            # [262956, 262956, 350608, 262956]
            
            start_column = 0
            end_column = 0
            list_attrib = ["xyz", "f_dc", "rotation", "scale"]
            attribues = {}
            numGauss = int(bp.dim_info[0]/3)
            for i, dim in enumerate(bp.dim_info):
                end_column += dim
                attribues.update({str(list_attrib[i]): deformed_gaussian[start_column:end_column].reshape(numGauss, -1)})
                start_column +=dim
            opacities = np.full((numGauss, 1), np.inf)
            gbp = GaussianBlenshapeProp(xyz = attribues["xyz"], f_dc= attribues["f_dc"], rotation=attribues["rotation"], scale = attribues["scale"], opacities=opacities, active_sh_degree=3)
            return gbp

    def get_NumGaussian(self):
        return self.get_xyz.shape[0]
    
    def get_nparray_xyz(self):
        points = []
        for i, point in enumerate(self._xyz.detach().cpu().numpy()):
            points.append([point[0], point[1], point[2]])
        
        # print("Number of points: ", len(points))

        points = np.array(points)

        return points
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def creat_from_hdf5(self, path_to_hdf5):
        f = h5py.File(path_to_hdf5, 'r')
        
        xyz = np.asarray(f['xyz'])
        xyz = torch.tensor(xyz).float().cuda()
        numVerts = xyz.shape[1]

        normals = np.asarray(f['normal'])
        normals = torch.tensor(normals).float().cuda()

        rgb = np.asarray(f['rgb'])
        fused_color = RGB2SH(torch.tensor(rgb).float().cuda())
        
        f_rest = np.asarray(f['f_rest'])
        f_rest = torch.tensor(f_rest).float().cuda()

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1)** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = f_rest

        scales = np.asarray(f['scale'])
        scales = torch.tensor(scales).float().cuda()

        rots = np.asarray(f['rotation'])
        rots = torch.tensor(rots).float().cuda()

        opacities = np.asarray(f['opacity'])
        opacities = torch.tensor(opacities).float().cuda()

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        # self._xyz = nn.Parameter(data = fused_point_cloud, requires_grad = False)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._normal = normals
        self.active_sh_degree = self.max_sh_degree

    def creat_from_GaussianBlendshapeProp(self, gbp:GaussianBlenshapeProp):
        numGauss = gbp.xyz.shape[0]

        print(f"SH degree: {self.max_sh_degree}")

        xyz = torch.tensor(gbp.xyz.reshape(numGauss, 3), dtype = torch.float, device = "cuda")
        print(f"xyz shape: {xyz.shape}")
        f_dc = torch.tensor(gbp.f_dc.reshape(numGauss, 3), dtype = torch.float, device = "cuda")
        print(f"f_dc shape: {f_dc.shape}")
        features = torch.zeros((numGauss, 3, (self.max_sh_degree + 1)** 2)).float().cuda()
        features[:, :3, 0] = f_dc
        features[:, 3:, 1:] = 0.0

        scales = torch.tensor(gbp.scale.reshape(numGauss, 3), dtype = torch.float, device = "cuda")

        rots = torch.tensor(gbp.rotation.reshape(numGauss, 4), dtype = torch.float, device = "cuda")

        opacities = np.full((numGauss, 1), np.inf, dtype=np.float16)
        opacities = torch.tensor(opacities, dtype = torch.float, device = "cuda")

        self._xyz = nn.Parameter(xyz.requires_grad_(False))
        # self._xyz = nn.Parameter(data = fused_point_cloud, requires_grad = False)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        print(f"feature_dc shape: {self._features_dc.shape}")
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        print(f"feature_rest shape: {self._features_rest.shape}")
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def update_from_GaussianBlendshapeProp(self, gbp:GaussianBlenshapeProp):
        numGauss = gbp.xyz.shape[0]
        xyz = torch.tensor(gbp.xyz.reshape(numGauss, 3), dtype = torch.float, device = "cuda")
        f_dc = torch.tensor(gbp.f_dc.reshape(numGauss, 3, 1), dtype = torch.float, device = "cuda")
        scales = torch.tensor(gbp.scale.reshape(numGauss, 3), dtype = torch.float, device = "cuda")
        rots = torch.tensor(gbp.rotation.reshape(numGauss, 4), dtype = torch.float, device = "cuda")
        opacities = torch.tensor(gbp.opacities.reshape(numGauss, 1), dtype = torch.float, device = "cuda")

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(f_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1

    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        print("initialize from point clouds (.ply) in directory")
        # self.lambda_lap = nn.Parameter(torch.tensor([0.5]).float().cuda()).requires_grad_(True)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # print(f"max_sh_degree: {self.max_sh_degree}")
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        normals = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # first_q_scale = (scales.min()+scales.mean())*0.5
        # scales = torch.ones_like(scales)*first_q_scale
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        rots[:, 1:] = normals

        # TODO: we assume opacity=1.0
        opacities = inverse_sigmoid(0.1*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        # TODO: Do not optimize center of Gaussians
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._xyz = nn.Parameter(data = fused_point_cloud, requires_grad = False)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._normal = nn.Parameter(normals.requires_grad_(True))
        # TODO: Do not optimize opacity (we assume opacity=1.0)
        self._opacity = nn.Parameter(data = opacities.requires_grad_(True), requires_grad=True)
        # self._opacity = nn.Parameter(opacities.requires_grad_(False), requires_grad=True)
        print(f"after init opacity.requires_grad: {self._opacity.requires_grad}")

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.parameters = l

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # self.xyz_grad_diff_args = get_expon_lr_func(lr_init=training_args.position_lr_init,
        #                                             lr_final=training_args.position_lr_final,
        #                                             max_steps=15_000)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=0.0000016,
                                                    lr_final=0.000000016,
                                                    # lr_delay_steps=7500,
                                                    # lr_delay_mult=0.1,
                                                    max_steps=15_000)

        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

        # self.opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_lr,
        #                                             lr_final=training_args.opacity_lr*1e-5,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

        # self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
        #                                             lr_final=training_args.scaling_lr*1e-5,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        # To check if xyz is fixed
        # for param_group in self.optimizer.param_groups:
        #     if param_group["name"] == "xyz":
        #         print(param_group)
        #         # param_group.requires_grad = False
        #         # print(f"xyz.required_grad: {param_group['params']['device']}")


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # elif param_group["name"] == "scaling":
            #     scale_lr = self.scaling_scheduler_args(iteration)
            #     param_group['lr'] = scale_lr
            #     return scale_lr
            # elif param_group["name"] == "opacity":
            #     opacity_lr = self.opacity_scheduler_args(iteration)
            #     param_group['lr'] = opacity_lr
            #     return opacity_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_GaussianProp(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        scaling_modifier=1.0
        L = build_scaling_rotation(scaling_modifier * self.get_scaling, self._rotation)
        tensor_covariance = L @ L.transpose(1, 2)
        covariance = tensor_covariance.detach().cpu().numpy()

        gs_prop = GaussianProp(xyz=xyz, normals=normals, f_dc=f_dc, f_rest=f_rest, opacities=opacities, scale=scale, rotation=rotation, covariance=covariance)
        dump_pckl(data=gs_prop, save_root=path, pickel_fname="gaussian_prop.pkl")


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        print("xyz",xyz.shape)
        normals = np.zeros_like(xyz)
        print("normals",normals.shape)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        print("f_dc",f_dc.shape)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        print("f_rest",f_rest.shape)
        opacities = self._opacity.detach().cpu().numpy()
        print("opacity",opacities.shape)
        scale = self._scaling.detach().cpu().numpy()
        print("scale", scale.shape)
        rotation = self._rotation.detach().cpu().numpy()
        print("rotation", rotation.shape)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # Where we resume training given model path
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # TODO: Do not optimize center of Gaussians
        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"), requires_grad=False)
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        # TODO: Do not optimize opacity (we assume opacity=1.0)
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"), requires_grad=False)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
