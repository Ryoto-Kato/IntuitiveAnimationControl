import h5py
import os, sys
import numpy as np
import trimesh
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

path_to_trained3dgs = os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", "trained_3dgs")
hdf5_87654 = "3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5"
hdf5_21954 = "upsampled_3dgs_21954_ALL_5perExp_trimesh_dcs.hdf5"
hdf5_5509 = "upsampled_3dgs_5509_ALL_5perExp_trimesh_dcs.hdf5"
path_to_hdf5_87652 = os.path.join(path_to_trained3dgs, hdf5_87654)
path_to_hdf5_21954 = os.path.join(path_to_trained3dgs, hdf5_21954)
path_to_hdf5_5509 = os.path.join(path_to_trained3dgs, hdf5_5509)
Ncomps = 330

f_87654 = h5py.File(path_to_hdf5_87652, "r")
DC_87654 = PCA_MBSPCA_SLDC(
    np.asarray(f_87654["dataMat"]),
    np.asarray(f_87654["pca_C"]).reshape(Ncomps, -1),
    np.asarray(f_87654["pca_W"]),
    np.asarray(f_87654["mbspca_C"]).reshape(Ncomps, -1),
    np.asarray(f_87654["mbspca_W"]),
    None, 
    None)

f_21954 = h5py.File(path_to_hdf5_21954, "r")
DC_21954 = PCA_MBSPCA_SLDC(
    np.asarray(f_21954["dataMat"]),
    np.asarray(f_21954["pca_C"]).reshape(Ncomps, -1),
    np.asarray(f_21954["pca_W"]),
    np.asarray(f_21954["mbspca_C"]).reshape(Ncomps, -1),
    np.asarray(f_21954["mbspca_W"]),
    None,
    None)

f_5509 = h5py.File(path_to_hdf5_5509, "r")
DC_5509 = PCA_MBSPCA_SLDC(
    np.asarray(f_5509["dataMat"]),
    np.asarray(f_5509["pca_C"]).reshape(Ncomps, -1),
    np.asarray(f_5509["pca_W"]),
    np.asarray(f_5509["mbspca_C"]).reshape(Ncomps, -1),
    np.asarray(f_5509["mbspca_W"]),
    None,
    None)


GTX = DC_87654.dataMat.copy()
print(GTX.shape)
print("DC_87652.pca_C", DC_87654.pca_C.shape)
print("DC_87652.pca_W", DC_87654.pca_W.shape)
# verts 87654
# RMSE w.r.t PCA
pca_87654_RMSE = np.sqrt(((GTX - np.matmul(DC_87654.pca_W[:, :Ncomps], DC_87654.pca_C[:Ncomps, :]))**2).mean())
pca_87654_sparsity = np.sum((DC_87654.pca_C[:Ncomps, :]**2).sum(axis = 1))
pca_87654_sparsity_level = np.mean(DC_87654.pca_C[:Ncomps, :] == 0)
print("-"*10)
print("verts 87654")
print("RMSE", pca_87654_RMSE)
print("Sparsity", pca_87654_sparsity)
print("Sparsity level", pca_87654_sparsity_level)

print("DC_21954.pca_C", DC_21954.pca_C.shape)
print("DC_21954.pca_W", DC_21954.pca_W.shape)

# verts 21954
# RMSE w.r.t PCA
pca_21954_RMSE = np.sqrt(((GTX - np.matmul(DC_21954.pca_W[:, :Ncomps], DC_21954.pca_C[:Ncomps, :]))**2).mean())
pca_21954_sparsity = np.sum((DC_21954.pca_C[:Ncomps, :]**2).sum(axis = 1))
pca_21954_sparsity_level = np.mean(DC_21954.pca_C[:Ncomps, :] == 0)
print("-"*10)
print("verts 21954")
print("RMSE", pca_21954_RMSE)
print("Sparsity", pca_21954_sparsity)
print("Sparsity level", pca_21954_sparsity_level)

print("DC_5509.pca_C", DC_5509.pca_C.shape)
print("DC_5509.pca_W", DC_5509.pca_W.shape)
# verts 87654
# RMSE w.r.t PCA
pca_5509_RMSE = np.sqrt(((GTX - np.matmul(DC_5509.pca_W[:, :Ncomps], DC_5509.pca_C[:Ncomps, :]))**2).mean())
pca_5509_sparsity = np.sum((DC_5509.pca_C[:Ncomps, :]**2).sum(axis = 1))
pca_5509_sparsity_level = np.mean(DC_5509.pca_C[:Ncomps, :] == 0)
print("-"*10)
print("verts 5509")
print("RMSE", pca_5509_RMSE)
print("Sparsity", pca_5509_sparsity)
print("Sparsity level", pca_5509_sparsity_level)
