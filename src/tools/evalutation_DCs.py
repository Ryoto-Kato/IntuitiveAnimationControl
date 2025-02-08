import h5py
import os, sys
import numpy as np
import trimesh
import pyvista as pv
from argparse import ArgumentParser, Namespace
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchSparsePCA
from dataclasses import dataclass

path_to_src = os.pardir
sys.path.append(path_to_src)
from utils.Blendshape import DeformationComponents
from utils.Blendshape import FaceMask
from utils.pickel_io import dump_pckl, load_from_memory
from utils.trackedmesh_loader import TrackedMeshLoader
from utils.vis_tools import VisPointsAttributes, VisPointsWithColor
from utils.converter import vector2MatNx3, vector3D2Scaler
from dreifus.matrix import Intrinsics, Pose
import matplotlib.pyplot as plt

def vis_influence_DCs(STD, Labels, title=""):
    for label,Std in zip(Labels, STD):
        sum_std = Std.max()
        df_std = Std/sum_std
        cumsum_std=df_std
        negative_cumsum_std = cumsum_std[:330]
        plt.plot(negative_cumsum_std, label=label)
    plt.ylabel("global influence of components")
    plt.xlabel("No. of principal components")
    plt.legend(Labels, loc="upper right")
    # plt.xticks(np.arange(0, len(Stds)-1, 5))
    plt.savefig(title+"Influence.png")

def vis_DCs(DCs, title=""):
    for DC in DCs:
        plt.plot(DC.ravel())
    plt.xlabel("ID of COG (the center of Gaussian)")
    plt.ylabel("The amount of deformation")
    plt.savefig(title+"DCs.png")

@dataclass
class PCA_MBSPCA_SLDC:
    dataMat: np.ndarray # centralized and normalized datamatrix
    pca_C: np.ndarray # not scaled yet
    pca_W:np.ndarray # not scaled yet
    mbspca_C: np.ndarray # not scaled yet
    mbspca_W: np.ndarray # not scaled yet
    sldc_C: np.ndarray
    sldc_W:np.ndarray

@dataclass
class EvaluatesResults_DCs:
    pca_RMSE: np.float16
    pca_sparsity: np.float16
    pca_sparsity_level: np.float16
    mbspca_RMSE: np.float16
    mbspca_sparsity: np.float16
    mbspca_sparsity_level: np.float16

parser = ArgumentParser(description="TrackedMesh and Trained3DGS")
path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples")
parser.add_argument('--path2trackedMeshFolder', type=str, default=os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", "tracked_mesh"))
parser.add_argument('--path2trained3dgsFolder', type=str, default=os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", "trained_3dgs"))
args = parser.parse_args(sys.argv[1:])

path_to_trackedMesh = args.path2trackedMeshFolder
path_to_hdf5_trackedMesh = os.path.join(path_to_trackedMesh, "trackedMesh_5509_ALL_5perExp_trimesh_dcs.hdf5")
path_to_trained3dgs = args.path2trained3dgsFolder
path_to_hdf5_trained3dgs = os.path.join(path_to_trained3dgs, "Final_3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5")
path_to_hdf5_onlyCOG_trained3dgs = os.path.join(path_to_trained3dgs, "Final_3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5")
path_to_hdf5_SLDC_trained_3dgs = os.path.join(path_to_trained3dgs, "Final_Best_gauss_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs_1.0_1.0_10.0.hdf5")
path_to_hdf5_OriginalSLDC_trained3dgs = os.path.join(path_to_trained3dgs, "Final_Baseline_original_SLDC_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs_2.0.hdf5")

f_trackedMesh = h5py.File(path_to_hdf5_trackedMesh, "r")
DC_trackedMesh = PCA_MBSPCA_SLDC(
    np.asarray(f_trackedMesh["dataMat"]),
    np.asarray(f_trackedMesh["pca_C"]),
    np.asarray(f_trackedMesh["pca_W"]),
    np.asarray(f_trackedMesh["mbspca_C"]),
    np.asarray(f_trackedMesh["mbspca_W"]),
    None, 
    None)

f_trained3dgs = h5py.File(path_to_hdf5_trained3dgs, "r")
DC_trained3dgs = PCA_MBSPCA_SLDC(
    np.asarray(f_trained3dgs["dataMat"]),
    np.asarray(f_trained3dgs["pca_C"]),
    np.asarray(f_trained3dgs["pca_W"]),
    np.asarray(f_trained3dgs["mbspca_C"]),
    np.asarray(f_trained3dgs["mbspca_W"]),
    None,
    None)

f_trained3dgs_onlyCOG = h5py.File(path_to_hdf5_onlyCOG_trained3dgs, "r")
DC_trained3dgs_onlyCOG = PCA_MBSPCA_SLDC(
    np.asarray(f_trained3dgs_onlyCOG["dataMat"]),
    np.asarray(f_trained3dgs_onlyCOG["pca_C"]),
    np.asarray(f_trained3dgs_onlyCOG["pca_W"]),
    np.asarray(f_trained3dgs_onlyCOG["mbspca_C"]),
    np.asarray(f_trained3dgs_onlyCOG["mbspca_W"]),
    None,
    None)

f_trained3dgs_SLDC = h5py.File(path_to_hdf5_SLDC_trained_3dgs, "r")
SLDC_trained3dgs = PCA_MBSPCA_SLDC(
    np.asarray(f_trained3dgs_SLDC["dataMat"]),
    None,
    None,
    None,
    None,
    np.asarray(f_trained3dgs_SLDC["sldc_C"]),
    np.asarray(f_trained3dgs_SLDC["sldc_W"]),
)

f_trained3dgs_OriginalSLDC = h5py.File(path_to_hdf5_OriginalSLDC_trained3dgs, "r")
Original_SLDC_trained3dgs = PCA_MBSPCA_SLDC(
    np.asarray(f_trained3dgs_OriginalSLDC["dataMat"]),
    None,
    None,
    None,
    None,
    np.asarray(f_trained3dgs_OriginalSLDC["sldc_C"]),
    np.asarray(f_trained3dgs_OriginalSLDC["sldc_W"]),
)

Ncomps = 330
Nverts = 5509

assert DC_trackedMesh.dataMat.shape == DC_trained3dgs.dataMat[:, :Nverts*3].shape

pca_STDs = []
pca_labels = []

GTX = DC_trackedMesh.dataMat.copy()
# tracked mesh
# RMSE w.r.t PCA
tm_pca_VAR = DC_trackedMesh.pca_W.diagonal()
tm_pca_STD = np.sqrt(tm_pca_VAR)
pca_STDs.append(tm_pca_STD)
pca_labels.append("Multi-face tracked mesh")
tm_pca_RMSE = np.sqrt(((GTX - np.matmul(DC_trackedMesh.pca_W[:, :Ncomps], DC_trackedMesh.pca_C[:Ncomps, :]))**2).mean())
tm_pca_sparsity = np.sum((DC_trackedMesh.pca_C[:Ncomps, :]**2).sum(axis = 1))
tm_pca_sparsity_level = np.mean(DC_trackedMesh.pca_C[:Ncomps, :] == 0)
print("-"*10)
print("tracked mesh PCA results")
print("RMSE", tm_pca_RMSE)
print("Sparsity", tm_pca_sparsity)
print("Sparsity level", tm_pca_sparsity_level)


# # RMSE w.r.t MBSPCA
# tm_mbspca_RMSE = np.sqrt(((GTX - np.matmul(DC_trackedMesh.mbspca_W[:, :Ncomps], DC_trackedMesh.mbspca_C[:Ncomps, :]))**2).mean())
# tm_mbspca_sparsity = np.sum((DC_trackedMesh.mbspca_C[:Ncomps, :]**2).sum(axis = 1))
# tm_mbspca_sparsity_level = np.mean(DC_trackedMesh.mbspca_C[:Ncomps, :] == 0)
# print("-"*10)
# print("tracked mesh MBSPCA results")
# print("RMSE", tm_mbspca_RMSE)
# print("Sparsity",tm_mbspca_sparsity)
# print("Sparsity level",tm_mbspca_sparsity_level)

Nverts = 87652
GTX = DC_trained3dgs_onlyCOG.dataMat[:, :Nverts*3].copy()
print(GTX.shape)
gs_pca_onlyCOG_VAR = DC_trained3dgs_onlyCOG.pca_W.diagonal()
gs_pca_onlyCOG_STD = np.sqrt(gs_pca_onlyCOG_VAR)
pca_STDs.append(gs_pca_onlyCOG_STD)
pca_labels.append("COG-PCA")
gs_pca_onlyCOG_RMSE = np.sqrt(((GTX - np.matmul(DC_trained3dgs_onlyCOG.pca_W[:, :Ncomps], DC_trained3dgs_onlyCOG.pca_C[:Ncomps, :Nverts*3]))**2).mean())
gs_pca_onlyCOG_sparsity = np.sum((DC_trained3dgs_onlyCOG.pca_C[:Ncomps, :Nverts*3]**2).sum(axis = 1))
gs_pca_onlyCOG_sparsity_level = np.mean(DC_trained3dgs_onlyCOG.pca_C[:Ncomps, :Nverts*3] == 0)

print("-"*10)
print("COG-PCA")
print("RMSE", gs_pca_onlyCOG_RMSE)
print("Sparsity",gs_pca_onlyCOG_sparsity)
print("Sparsity level",gs_pca_onlyCOG_sparsity_level)


GTX = DC_trained3dgs.dataMat[:, :Nverts*3].copy() #only vertex parts
# trained 3dgs
# RMSE w.r.t PCA
gs_pca_VAR= DC_trained3dgs.pca_W.diagonal()
gs_pca_STD = np.sqrt(gs_pca_VAR)
pca_STDs.append(gs_pca_STD)
pca_labels.append("Ours (global)")
gs_pca_RMSE = np.sqrt(((GTX - np.matmul(DC_trained3dgs.pca_W[:, :Ncomps], DC_trained3dgs.pca_C[:Ncomps, :Nverts*3]))**2).mean())
gs_pca_sparsity = np.sum((DC_trained3dgs.pca_C[:Ncomps, :Nverts*3]**2).sum(axis = 1))
gs_pca_sparsity_level = np.mean(DC_trained3dgs.pca_C[:Ncomps, :Nverts*3] == 0)

print("-"*10)
print("Ours (global)")
print("RMSE", gs_pca_RMSE)
print("Sparsity",gs_pca_sparsity)
print("Sparsity level",gs_pca_sparsity_level)

# # RMSE w.r.t MBSPCA
# gs_mbspca_RMSE = np.sqrt(((GTX - np.matmul(DC_trained3dgs.mbspca_W[:, :Ncomps], DC_trained3dgs.mbspca_C[:Ncomps, :Nverts*3]))**2).mean())
# gs_mbspca_sparsity = np.sum((DC_trained3dgs.mbspca_C[:Ncomps, :Nverts*3]**2).sum(axis = 1))
# gs_mbspca_sparsity_level = np.mean(DC_trained3dgs.mbspca_C[:Ncomps, :Nverts*3] == 0)

# print("-"*10)
# print("trained 3dgs MBSPCA results")
# print("RMSE", gs_mbspca_RMSE)
# print("Sparsity",gs_mbspca_sparsity)
# print("Sparsity level",gs_mbspca_sparsity_level)

Ncomps = 330

SLDC_trained3dgs.dataMat = SLDC_trained3dgs.dataMat.reshape(SLDC_trained3dgs.dataMat.shape[0], -1)
# GTX = SLDC_trained3dgs.dataMat[:, :Nverts*3].copy()# only vertex parts
SLDC_trained3dgs.sldc_C = SLDC_trained3dgs.sldc_C.reshape(SLDC_trained3dgs.sldc_C.shape[0], -1)
# trained 3dgs
# RMSE w.r.t SLDC
print(SLDC_trained3dgs.sldc_W[:, :Ncomps].shape)
print(SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3].shape)
gs_sldc_RMSE = np.sqrt(((GTX - np.matmul(SLDC_trained3dgs.sldc_W[:, :Ncomps], SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3]))**2).mean())
gs_sldc_sparsity = np.sum((SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3]**2).sum(axis = 1))
gs_sldc_sparsity_level = np.mean(SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3] == 0)


print("-"*10)
print("Ours (local)")
print("RMSE", gs_sldc_RMSE)
print("Sparsity",gs_sldc_sparsity)
print("Sparsity level", gs_sldc_sparsity_level)

Original_SLDC_trained3dgs.dataMat = Original_SLDC_trained3dgs.dataMat.reshape(Original_SLDC_trained3dgs.dataMat.shape[0], -1)
# GTX = Original_SLDC_trained3dgs.dataMat[:, :Nverts*3].copy()# only vertex parts
Original_SLDC_trained3dgs.sldc_C = Original_SLDC_trained3dgs.sldc_C.reshape(Original_SLDC_trained3dgs.sldc_C.shape[0], -1)
# trained 3dgs
# RMSE w.r.t SLDC
print(Original_SLDC_trained3dgs.sldc_W[:, :Ncomps].shape)
print(Original_SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3].shape)
gs_original_sldc_RMSE = np.sqrt(((GTX - np.matmul(Original_SLDC_trained3dgs.sldc_W[:, :Ncomps], Original_SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3]))**2).mean())
gs_original_sldc_sparsity = np.sum((Original_SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3]**2).sum(axis = 1))
gs_original_sldc_sparsity_level = np.mean(Original_SLDC_trained3dgs.sldc_C[:Ncomps, :Nverts*3] == 0)


print("-"*10)
print("Trapezoidal SLDC")
print("RMSE", gs_original_sldc_RMSE)
print("Sparsity",gs_original_sldc_sparsity)
print("Sparsity level", gs_original_sldc_sparsity_level)

# vis_DCs(DCs = SLDC_trained3dgs.sldc_C[:1, :], title="./SLDC/SLDC_DC")
# vis_DCs(DCs = Original_SLDC_trained3dgs.sldc_C[:1, :], title="./OriginalSLDC/OriginalSLDC_DC")

average_face = pv.read(os.path.join(path_to_sample, "3dgs", "sample_subd2_face.ply"))

intrinsic = Intrinsics(np.asarray([[7717.2456, 0.0, 618.5467],
                        [0.0, 7717.927, 899.3488],
                        [0.0, 0.0, 1.0]]))

extrinsic = Pose(np.asarray([[0.9994892, 0.014108674, 0.028675145, -20.131886],
                        [-0.0048997356, 0.95431656, -0.29875728, 309.65524],
                        [-0.031580236, 0.29846418, 0.95389825, 46.042873],
                        [0.0, 0.0, 0.0, 1.0]]))
window_size = [1334, 2048]
print("camera extrinsic shape: ", extrinsic.shape)
print("camera intrinsic shape: ", intrinsic.shape)

#[TODO]
path_to_output=os.path.join(os.getcwd(), os.pardir, "output")
os.mkdir(path_to_output, "OursG")
os.mkdir(path_to_output, "COG-PCA")
os.mkdir(path_to_output, "OursL")
os.mkdir(path_to_output, "T-SLDC")


for i in range(10):
    deformation_pca = vector3D2Scaler(vector=DC_trained3dgs.pca_C[i:i+1, :Nverts*3].ravel(), num_verts=Nverts, normalized=True)
    deformation_onlyCOG_pca = vector3D2Scaler(vector=DC_trained3dgs_onlyCOG.pca_C[i:i+1, :].ravel(), num_verts=Nverts, normalized=True)
    deformation_SLDC = vector3D2Scaler(vector=SLDC_trained3dgs.sldc_C[i:i+1, :].ravel(), num_verts=Nverts, normalized=True)
    deformation_Original_SLDC = vector3D2Scaler(vector=Original_SLDC_trained3dgs.sldc_C[i:i+1, :].ravel(), num_verts=Nverts, normalized = True)
    VisPointsAttributes(points= average_face.points, attributes = deformation_pca, screenshot=True, title = "../output/OursG/"+str(i), flag_render_from_camera=True, window_size=window_size, intrinsic=intrinsic, extrinsic=extrinsic)
    VisPointsAttributes(points= average_face.points, attributes = deformation_onlyCOG_pca, screenshot=True, title = "../output/COG-PCA/"+str(i), flag_render_from_camera=True, window_size=window_size, intrinsic=intrinsic, extrinsic=extrinsic)
    VisPointsAttributes(points= average_face.points, attributes = deformation_SLDC, screenshot=True, title = "../output/OursL/"+str(i), flag_render_from_camera=True,window_size=window_size, intrinsic=intrinsic, extrinsic=extrinsic)
    VisPointsAttributes(points= average_face.points, attributes = deformation_Original_SLDC, screenshot=True, title = "../output/T-SLDC/"+str(i), flag_render_from_camera=True,window_size=window_size, intrinsic=intrinsic, extrinsic=extrinsic)

vis_influence_DCs(STD = pca_STDs, Labels=pca_labels, title="Comparison_influence_DCs")