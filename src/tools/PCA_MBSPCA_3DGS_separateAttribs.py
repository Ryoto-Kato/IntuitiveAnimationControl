import h5py
import os, sys
import numpy as np
import trimesh
from argparse import ArgumentParser, Namespace
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchSparsePCA

path_to_src = os.pardir
sys.path.append(path_to_src)
from utils.Blendshape import DeformationComponents
from utils.Blendshape import FaceMask
from utils.pickel_io import dump_pckl, load_from_memory
from utils.trackedmesh_loader import TrackedMeshLoader

'''
Execution sample
    conda activate 3dsrf
    python PCA_MBSPCA_3DGS_separateAttribs.py --hdf5_fname="f336a291-bnotALLcam_datamat_87652.hdf5"
'''

parser = ArgumentParser(description="PCA and MBSPCA")
parser.add_argument('--hdf5_fname', type = str, default=None)
parser.add_argument('--faceMask', action = 'store_true', default = False)
parser.add_argument('--Ncompos', type = int, default = 330)
args = parser.parse_args(sys.argv[1:])

hdf5_fname = args.hdf5_fname
with_faceMask = args.faceMask
Ncompos = args.Ncompos

datasetName = "3dgs"
path_to_dataset=os.path.join(os.getcwd(), os.pardir, "samples", 'deformation_components', 'trained_3dgs')
path_to_hdf5 = os.path.join(path_to_dataset, args.hdf5_fname)
assert os.path.exists(path_to_hdf5)

path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples", "3dgs")

# reading hdf5
print(f"Reading hdf5: {path_to_hdf5}")
f = h5py.File(os.path.join(path_to_hdf5), 'r')

print(f"Contents of the hdf5: {path_to_hdf5}")
list_attribs = list(f.keys())

numVerts = int((np.asarray(f['xyz']).shape[1])/3)
numSamples = int((np.asarray(f['xyz']).shape[0]))
print(f"The number of verts: {numVerts}")
print(f"The number of samples: {numSamples}")

# standardization
dataMat = np.asarray(f['xyz']).reshape(numSamples, -1)
# MEAN = np.mean(dataMat,axis = 0)
MEAN = dataMat[0]
STD = np.full_like(MEAN, np.std(dataMat), dtype = np.float64)
# print(STD)
dataMat = (dataMat - MEAN[None, :])/STD[None, :]

dim_info= []
list_attrName = []
list_attrName.append("xyz")
dim_info.append(dataMat.shape[1])

dataMats = []
dataMats.append(dataMat)

outputAttr_names=[]

for i, key in enumerate(list_attribs):
    if key == "rgb" or key == "xyz" or key == "normal" or key == "opacity" or key == "f_rest":
        continue
    print(f"{i}:{key}")
    list_attrName.append(key)
    new_attributes = np.asarray(f[key])
    new_attributes = new_attributes.reshape(numSamples, -1)
    
    # mean_new_attribs = np.mean(new_attributes, axis = 0)
    mean_new_attribs = new_attributes[0]
    std_new_attribs = np.full_like(mean_new_attribs, np.std(new_attributes), dtype = np.float64)

    # standardization
    new_attributes = (new_attributes - mean_new_attribs[None, :])/std_new_attribs[None, :]

    MEAN = np.concatenate((MEAN, mean_new_attribs), axis = 0)

    STD = np.concatenate((STD, std_new_attribs), axis = 0)
    
    dim_info.append(new_attributes.shape[1])
    dataMats.append(new_attributes)
    print(new_attributes.shape)

list_keys = list_attrName

print("Mean: ", MEAN.shape)
print("STD: ", STD.shape)

DataMat_allatrib = None
for j, dM in enumerate(dataMats):
    if j == 0:
        DataMat_allatrib = dM
    else:
        DataMat_allatrib = np.concatenate((DataMat_allatrib, dM), axis = 1)

print("Data mat for all attributes: ", DataMat_allatrib.shape)

print("list of attribute in dataMat")
print(list_attrName)
print("list of dimension in dataMat for each attribute")
print(dim_info)

if numVerts == 5509:
    mesh = trimesh.load(os.path.join(path_to_sample, "sample_face.ply"), force = 'mesh')
    tris = np.asarray(mesh.faces)
elif numVerts == 87652:
    mesh = trimesh.load(os.path.join(path_to_sample, "sample_subd2_face.ply"), force = 'mesh')
    tris = np.asarray(mesh.faces)

# print(dim_info)

# loading facemask
path_to_facemaskPKL = os.path.join(os.getcwd(), os.pardir, "samples", "memory")
if numVerts == 5509:
    pickel_fname = "FaceMask_sample_face_22012024_13_25_trimesh.pkl"
elif numVerts == 87652:
    pickel_fname = "FaceMask_sample_subd2_face_22012024_12_51_trimesh.pkl"
assert os.path.exists(os.path.join(path_to_facemaskPKL,pickel_fname))
print(f"loading facemask from {path_to_facemaskPKL}")

facemask = load_from_memory(path_to_memory=path_to_facemaskPKL, pickle_fname=pickel_fname)
print(f"facemask shape: {facemask.bit_mask.shape}")

DCs = {}

for i, (key, dataMat) in enumerate(zip(list_keys, dataMats)):
    print(f"-------------{key}-------------------")
    shape_dataMat = dataMat.shape
    print(shape_dataMat)
    if with_faceMask:
        masked_GTX = dataMat * facemask.bit_mask[None, :]
        GTX = masked_GTX  #masked_N_cent_X
    else:
        GTX = dataMat

    # PCA on full samples
    D = GTX.shape[0]
    Ncompos = D
    print(f"The number of components: {Ncompos}")
    pca = PCA(D)
    pca.fit(GTX)

    Gamma = pca.components_.reshape(dataMat.shape)
    Sigma = np.diag(pca.explained_variance_)
    print(f"Sigma shape: {Sigma[:, :Ncompos].shape}")
    print(f"Gamma shape: {Gamma[:Ncompos, :].shape}")

    # # MBSPCA for Ncompos samples

    if key != "xyz":
        W_mbspca = None
        C_mbspca = None
    else:
        mb_sparsePCA = MiniBatchSparsePCA(n_components=Ncompos, verbose=False)
        est_MBSPCA = mb_sparsePCA.fit(GTX)

        C_mbspca = est_MBSPCA.components_.reshape(Ncompos, -1) #right hand side V
        if with_faceMask:
            W_mbspca = est_MBSPCA.transform(masked_GTX.reshape(masked_GTX.shape[0], -1)) #left hand side U
        else:
            W_mbspca = est_MBSPCA.transform(GTX) #left hand side U
        print(f"W shape: {W_mbspca.shape}")
        print(f"C shape: {C_mbspca.shape}")

    DCs.update({key: {"dataMat": DataMat_allatrib, "faceMask": facemask.bit_mask, "MEAN": MEAN, "STD": STD, "pca_W": Sigma, "pca_C": Gamma, "mbspca_W": W_mbspca, "mbspca_C": C_mbspca, "tris": tris, "dim_info": Gamma.shape[1]}})

# print(PCAs)
# print(MBSPCAs)
print(DCs.keys())

for i, dc_key in enumerate(DCs.keys()):
    hdf5_saveName = datasetName+"_"+str(numVerts)+"_"+str(dc_key)+"_"+"PCAMBSPCA"+"_"+"5perExp_trimesh_dcs.hdf5"

    print(f"Saving PCA and MBSCPA at {os.path.join(path_to_dataset, hdf5_saveName)}")
    f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName), 'w')

    dset = f.create_dataset(name = "dim_info", data=np.asarray(DCs[dc_key]["dim_info"]))
    dset.attrs["list_attrName"] = str(list_attrName)
    dset1 = f.create_dataset(name = "dataMat", data=DCs[dc_key]["dataMat"])
    dset2 = f.create_dataset(name = "faceMask", data=DCs[dc_key]["faceMask"])
    dset3 = f.create_dataset(name = "MEAN", data = DCs[dc_key]["MEAN"])
    dset4 = f.create_dataset(name = "STD", data = DCs[dc_key]["STD"])
    dset5 = f.create_dataset(name = "pca_W", data = DCs[dc_key]["pca_W"])
    dset6 = f.create_dataset(name = "pca_C", data = DCs[dc_key]["pca_C"])
    if dc_key == "xyz":
        dset7 = f.create_dataset(name = "mbspca_W", data = DCs[dc_key]["mbspca_W"])
        dset8 = f.create_dataset(name = "mbspca_C", data = DCs[dc_key]["mbspca_C"])
    dset9 = f.create_dataset(name = "tris", data = DCs[dc_key]["tris"])

    # check if they are actually stored in hdf5
    # out_f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName))
    # print(list(out_f.keys()))
    # print(np.asarray(out_f["dataMat"]).shape)
    # print(np.asarray(out_f["MEAN"]).shape)
    