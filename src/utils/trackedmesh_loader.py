import os
import sys
import trimesh
import numpy as np

from utils.Dataset_handler import Filehandler
from utils.OBJ_helper import OBJ


"""
Dataset loader only for this project to obtain data matrix X
This script represents scripts on SLDC.ipynb before "face mask"

"""

class TrackedMeshLoader:
    """
    Arguments
        path_to_dataset:    str
        ID:                 int
        suffix:             str (the suffix of mesh_loaderthe expression folder)
        mesh_loader:        str (type of mesh [trimesh or original])
        num_samples_perExp: int 
    """
    def __init__(self, path_to_dataset, ID = 6795937, suffix = 'E0', mesh_loader = "trimesh", num_samples_perExp = 5):
        self.path_to_dataset = path_to_dataset
        self.ID = ID
        self.suffix = suffix
        self.mesh_loader = mesh_loader
        self.num_samples_perExp = num_samples_perExp
        self.list_exps_name = [] #list of expression name
        self.map_exps2id = {} #mapping from expression name to id
        self.file_handler = None
        self.dict_expMeshes = {} # dictionary to contain meshes
        self.dict_expVerts = {} # dictionary to contain vertices

        self.num_vertices = None
        self.len_col = None

        self.X = None
        self.ave_neutralmesh_vertices = None
        self.cent_X = None
        self.centX_std = None
        
    def __call__(self):
        # create map from exp name to id
        self.create_ListOfExp()

        # Get file path
        self.file_handler = Filehandler(path_to_dataset=self.path_to_dataset)
        self.file_handler.iter_dir()

        # Load obj file by using selected mesh loader
        print("Loading meshes....")
        for expID, key in enumerate(self.file_handler.dict_objs.keys()):
            list_Meshes = []
            list_Verts = []

            for i, obj in enumerate(self.file_handler.dict_objs[key][0:self.num_samples_perExp]):
                path_to_obj = os.path.join(self.file_handler.list_expPathFiles[expID], obj)

                if self.mesh_loader == "trimesh":
                    _mesh = trimesh.load(path_to_obj, force = 'mesh')
                elif self.mesh_loader == "original":
                    _mesh = OBJ(path_to_obj, swapyz=False)
                
                list_Meshes.append(_mesh)
                list_Verts.append(_mesh.vertices)

            self.dict_expMeshes.update({expID: list_Meshes})
            self.dict_expVerts.update({expID: list_Verts})
        self.num_vertices = len(self.dict_expVerts[0][0])
        self.len_col = self.num_vertices * 3

        _list_xs = []
        num_sum_samples = 0

        for key in self.dict_expVerts.keys():
            vertices = self.dict_expVerts[key]
            _num_samples = len(vertices)

            num_sum_samples = num_sum_samples + _num_samples

            _array = np.array(vertices)
            _list_xs.append(_array)

        print("Construction data matrix X....")
        
        neutralmesh_verts = _list_xs[0]

        self.X = _list_xs[0]

        for x in _list_xs[1:]:
            self.X = np.concatenate((self.X, x), axis=0)

        self.ave_neutralmesh_vertices = np.mean(neutralmesh_verts, axis=0)
        self.cent_X = self.X - self.ave_neutralmesh_vertices[None, :]
        self.centX_std = np.std(self.cent_X)

        print("done")

        return self.X, self.ave_neutralmesh_vertices, self.cent_X

        
    def create_ListOfExp(self):
        """
        Returns
            update self.list_exps_name and self.map_exp2id
        """
        counter = 0
        # get expression name which are written as folder name and starting with given suffix
        for i, name in enumerate(os.listdir(self.path_to_dataset)):
            f = os.path.join(self.path_to_dataset, name)
            if os.path.isdir(f) and name.startswith(self.suffix):
                counter = counter + 1
                self.list_exps_name.append(name)
        
        # sort list of expression name
        self.list_exps_name.sort()

        # create mapping from expression name and id
        for i, exp_name in enumerate(self.list_exps_name):
            self.map_exps2id.update({exp_name: i})

    
    

