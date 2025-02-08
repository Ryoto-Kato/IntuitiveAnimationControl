import numpy as np
from .OBJ_helper import OBJ
from dataclasses import dataclass
import os
import h5py

@dataclass
class ZeroMeanDefMatrix:
    masked_cent_x: np.ndarray
    mean: np.ndarray
    std: float
    tris: np.ndarray

@dataclass
class FaceMask:
    center_id: int
    list_ids: list
    bit_mask: np.ndarray

@dataclass
class datastruct_blendshape:
    """
    K: the number of componnets
    N: the number of vertex

    Arguments
        MEAN: shape [N*3,]
        PCs: shape [K, N, 3]
        Stds: shape [K, ]
    """
    ID: int
    List_exps: list
    MEAN: np.ndarray
    PCs: np.ndarray
    Stds: np.ndarray

class DeformationComponents:
    """
    K: the number of components = 50
    V: the number of vertex = 5509
    N: the number of samples (data) = 330
    
    Arguments
        dataMat: shape[N, V, 3]
        facemask (bitmask): shape[1, V, 3]
        MEAN: shape[1, 3*V]
        STD: float
        pcMat: 
            if method is PCA: shape[N, V, 3]
            if method is MBSPCA: shape[K, V, 3]

        coeffMat: 
            if method is PCA: shape[N, N]
            if method is MBSPCA: shape[N, K]

        tris: triangle list [#triangle, 3]

        NofExP: the number of expressions
        NofFrame: the number of frames per expression
        NofVerts: the number of vertex (=V)
    """
    def __init__(self, dataMat:np.ndarray, faceMask:FaceMask, MEAN:np.ndarray, STD, pcMat:np.ndarray, coeffMat:np.ndarray, tris:np.ndarray, NofFrame:int, NofExp:int, NofVerts:int):
        self.dataMat = dataMat
        self.faceMask = faceMask
        self.MEAN = MEAN
        self.STD = STD
        self.pcMat = pcMat
        self.coeffMat = coeffMat
        self.tris = tris
        self.NofExp = NofExp
        self.NofFrame = NofFrame
        self.NofVerts = NofVerts

    def save_hdf5(self, path_to_save, fname):
        assert os.path.exists(path_to_save)
        with h5py.File(os.path.join(path_to_save, fname), "w") as f:
            dset = f.create_dataset(name = "dataMat", data=self.dataMat)
            dset = f.create_dataset(name = "faceMask", data=self.faceMask)
            dset = f.create_dataset(name = "MEAN", data = self.MEAN)
            dset = f.create_dataset(name = "STD", data = self.STD)
            dset = f.create_dataset(name = "pcMat", data = self.pcMat)
            dset = f.create_dataset(name = "coeffMat", data = self.coeffMat)
            dset = f.create_dataset(name = "tris", data = self.tris)
            dset = f.create_dataset(name = "NofExp", data = self.NofExp)
            dset = f.create_dataset(name = "NofFrame", data = self.NofFrame)
            dset = f.create_dataset(name = "NofVerts", data = self.NofVerts)
        print(f"save hdf5 at {os.path.join(path_to_save, fname)}")

class Blendshape:
    def __init__(self, Verts_averageExp, PCs, Stds, D:int, save_path, only_specific_pc:bool, name_newExp = ""):
        # Verts_averageExp: (np.darray) vertex coordinates of average expression of specific ID
        # PCs: (Matrix, shape=[#blendshapes, #vertices]) Principal components 
        # Stds: (np.darray, shape=[#blendshapes]) standard deviation
        # D: (int, >0) desired dimensionality of blendshapes, or in the case `only_specific_pc` == true, D should be the id of the principle component which you want to use for generation of new expression

        self.averageExp = Verts_averageExp
        self.PCs = PCs
        self.Stds = Stds
        self.given_PCs_dim = PCs.shape[0]
        self.coefficients = np.zeros(D)
        self.newFace_vertices = None
        # Sample_path
            # Since tracked meshes are topologically equivalent, we can make use of the part for vt, vn, and f in a .obj from the dataset
            # This sample path should be the path to sample.obj which is duplicated from abitrary obj file from a tracked mesh folder 
        self.sample_path = os.path.join(os.getcwd(), os.pardir, 'samples', 'trimesh_sample.obj') #trimesh_sample.obj
        self.save_path = save_path
        self.name_newExp = name_newExp
        self.D = D
        if (self.D <1):
            print("D should be unsigned integer")            
        else:
            self.D = D

        if only_specific_pc:
            self.start_id_pcs = D-1
        else:
            self.start_id_pcs = 0
        # self.sample_coefficients()
        # self.get_newExp()
        # _ = self.generate_newExp()
    
    def sample_coefficients(self):
        coefficients = []
        for i, std in enumerate(self.Stds[self.start_id_pcs:self.D]):
            mu,sigma = 0, std # mean and standard deviation
            _noise = np.random.normal(mu, sigma, 1)
            coefficients.append(_noise)
        self.coefficients = coefficients

    def set_individual_coefficient(self, index, value, debug):
        if debug:
            print(f"index: {index}")
            print(f"value: {value}")
        self.coefficients[index] = value

    def print_coefficients(self):
        print(self.coefficients)
        print(len(self.coefficients))

    def set_coefficients(coeffcients):
        self.coefficients = coeffcients

    def get_newExp(self):
        newFace_vertices = self.averageExp
        for i, coeff in enumerate(self.coefficients):
            _item = coeff*self.PCs[i]
            newFace_vertices = newFace_vertices + _item
        self.newFace_vertices = newFace_vertices
    
    def generate_newExp(self):
        return OBJ.write_OBJfile(reference_obj_file= self.sample_path, save_path = self.save_path, vertices =self.newFace_vertices, name_Exp=self.name_newExp)

    def generate_aveExp(self):
        return OBJ.write_OBJfile(reference_obj_file= self.sample_path, save_path = self.save_path, vertices =self.averageExp, name_Exp="averageExp")