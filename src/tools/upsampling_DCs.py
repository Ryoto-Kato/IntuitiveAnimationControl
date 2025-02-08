import h5py
import os
import sys
import numpy as np
sys.path.append(os.pardir)
from utils.DeformationComponents_utils import PCA_MBSPCA_SLDC, attribute_subd1, attribute_subd2
import trimesh
from argparse import ArgumentParser, Namespace
from sklearn.decomposition import PCA


def upsampling_PCAMBSPCA_selectedAttribs(args):
    Nverts = args.numGauss

    if Nverts == 21954:
        hdf_DC_fname = "3dgs_21954_ALL_5perExp_trimesh_dcs.hdf5"
    elif Nverts == 5509:
        hdf_DC_fname = "3dgs_5509_ALL_5perExp_trimesh_dcs.hdf5"
    else:
        print("you need to feed --numGauss which has corresponding hdf5 file")

    path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples", "3dgs")
    path_to_trained3dgs = os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", "trained_3dgs")
    path_to_hdf5_trained3dgs = os.path.join(path_to_trained3dgs, hdf_DC_fname)
    sample_21954_mesh = trimesh.load(os.path.join(path_to_sample, "sample_subd_face.ply"), force='mesh')
    sample_5509_mesh = trimesh.load(os.path.join(path_to_sample, "sample_face.ply"), force='mesh')

    f_trained3dgs = h5py.File(path_to_hdf5_trained3dgs, "r")
    DC_trained3dgs = PCA_MBSPCA_SLDC(
        np.asarray(f_trained3dgs["dataMat"]),
        np.asarray(f_trained3dgs["pca_C"]),
        np.asarray(f_trained3dgs["pca_W"]),
        np.asarray(f_trained3dgs["mbspca_C"]),
        np.asarray(f_trained3dgs["mbspca_W"]),
        None,
        None)


    W = DC_trained3dgs.pca_W
    Ncomps = W.shape[0]
    C = DC_trained3dgs.pca_C.reshape(Ncomps, Nverts, -1)

    print(f"pca_C shape {C.shape}")
    print(f"pca_W shape {W.shape}")


    temp_C = C.copy()
    print(temp_C.shape)
    reshaped_C = temp_C[0]
    if Nverts == 5509:
        sample_mesh = sample_5509_mesh
        sample_faces = sample_5509_mesh.faces
    elif Nverts == 21954:
        sample_mesh = sample_21954_mesh
        sample_faces = sample_21954_mesh.faces

    final_C = None
    for i in range(0, temp_C.shape[0]):
        print("-"*5 + str(i) + "-"*5)
        target_column = temp_C[i].reshape(Nverts, -1)
        # print(f"target_clolumn shape{target_column.shape}")
        target_attribs = target_column[:, 3:]
        # print(f"target_attribs shape{target_attribs.shape}")
        target_verts = target_column[:, :3]
        # print(f"target_verts shape {target_verts.shape}")
        
        # ["xyz", "f_dc", "rotation", "scale"]

        C_xyz, _ = attribute_subd2(attribute=target_column[:, :3], sample_mesh=sample_mesh)
        # C_xyz = C_xyz/np.max(C_xyz, axis=0)[None, :]
        C_subd2 = C_xyz
        C_f_dc, _ = attribute_subd2(attribute=target_column[:, 3:6], sample_mesh=sample_mesh)
        # C_f_dc = np.zeros_like(C_xyz)
        C_subd2 = np.concatenate((C_subd2, C_f_dc), axis = 1)
        # C_rot = np.zeros((C_subd2.shape[0], 4))
        C_rot, _ = attribute_subd2(attribute=target_column[:, 6:10], sample_mesh=sample_mesh)
        C_subd2 = np.concatenate((C_subd2, C_rot), axis = 1)
        # C_scale = np.zeros_like(C_xyz)
        C_scale, _ = attribute_subd2(attribute=target_column[:, 10:13], sample_mesh=sample_mesh)
        C_subd2 = np.concatenate((C_subd2, C_scale), axis = 1)
        # print(f"C_attribs shape: {C_attribs.shape}")
        # print(f"C_verts shape: {C_verts.shape}")
        print(f"upsampled attributes shape: {C_subd2.shape}")
        if i == 0:
            final_C = C_subd2.reshape(1, 87652, 13)
            print(f"first column in final_C shape {final_C.shape}")
        else:
            final_C = np.concatenate((final_C, C_subd2[None, :]),  axis = 0)
            print(f"final_C shape : {final_C.shape}")
    # print(f"reshape_C shape {reshaped_C.shape}")
    # if Nverts == 5509:
    #     C_subd2, _ = attribute_subd2(attribute=reshaped_C, sample_mesh = sample_5509_mesh)
    # elif Nverts == 21954:
    #     C_subd2, _ = attribute_subd1(attribute=reshaped_C, sample_mesh = sample_21954_mesh)
    # final_C = None
    # final_C = C_subd2[:, :13].reshape(1, 87652, 13)
    # print(f"final_C shape {final_C.shape}")
    # for i in range (1, temp_C.shape[0]):
    #     # print(f"final_C shape {final_C.shape}")
    #     final_C = np.concatenate((final_C, C_subd2[..., 13*i:13*(i+1)].reshape(87652, 13)[None, :]), axis = 0)
    final_C = final_C.reshape(final_C.shape[0], -1)

    # apply eigen decomposition to ensure that each vector is normalized
    GTX = np.matmul(W, final_C)
    print(f"W*C: {GTX.shape}")
    pca = PCA(Ncomps)
    pca.fit(GTX)

    new_C = pca.components_.reshape(GTX.shape)
    new_W = np.diag(pca.explained_variance_)
    print(f"new C shape: {new_C.shape}")
    print(f"new W shape: {new_W.shape}")

    print("previous W")
    print(W[0][0], W[1][1])
    print("new W")
    print(new_W[0][0], new_W[1][1])

    print(f"After concatenation final_C shape: {final_C.shape}")
    hdf5_saveName = "test_upsampled_"+hdf_DC_fname
    datasetName = "3dgs"

    # verts 5509
    # [xyz, f_dc, rotation, scale]
    # [16527, 16527, 22036, 16527]
    # verts 21954
    # [65862, 65862, 87816, 65862]
    # verts 87652
    # [xyz, f_dc, rotation, scale]
    # [262956, 262956, 350608, 262956]

    f = h5py.File(os.path.join(path_to_trained3dgs, hdf5_saveName), 'w')
    print(np.asarray(f_trained3dgs["STD"]).shape)
    print(np.asarray(f_trained3dgs["dim_info"]))

    list_attrName = ["xyz", "f_dc", "rotation", "scale"]
    dset = f.create_dataset(name = "dim_info", data=np.asarray(f_trained3dgs["dim_info"]))
    dset.attrs["list_attrName"] = str(list_attrName)
    dset1 = f.create_dataset(name = "dataMat", data=DC_trained3dgs.dataMat)
    dset3 = f.create_dataset(name = "MEAN", data = np.asarray(f_trained3dgs["MEAN"]))
    dset4 = f.create_dataset(name = "STD", data = np.asarray(f_trained3dgs["STD"]))
    # dset5 = f.create_dataset(name = "pca_W", data = DC_trained3dgs.pca_W)
    # dset6 = f.create_dataset(name = "pca_C", data = final_C)
    dset5 = f.create_dataset(name = "pca_W", data = new_W)
    dset6 = f.create_dataset(name = "pca_C", data = new_C)
    dset7 = f.create_dataset(name = "mbspca_W", data = DC_trained3dgs.mbspca_W)
    dset8 = f.create_dataset(name = "mbspca_C", data = DC_trained3dgs.mbspca_C)
    dset9 = f.create_dataset(name = "tris", data = np.asarray(f_trained3dgs["tris"], dtype=int))

    out_f = h5py.File(os.path.join(path_to_trained3dgs, hdf5_saveName), 'r')
    print(np.asarray(out_f['pca_C']).shape)


'''
Execution sample
    conda activate 3dsrf
    python upsampling_DCs.py --numGauss=5509
'''

if __name__ == "__main__":
    parser = ArgumentParser(description="Upsampling only available for PCA/MBSPCA")
    parser.add_argument('--numGauss', type = int, default=5509)
    args = parser.parse_args(sys.argv[1:])
    upsampling_PCAMBSPCA_selectedAttribs(args=args)