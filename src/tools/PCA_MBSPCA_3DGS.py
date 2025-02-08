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


def PCA_MBSPCA_selectedAttribsAtOnce(args):

    path_to_folder = args.path2folder
    path_to_hdf5 = os.path.join(args.path2folder, args.hdf5_fname)
    hdf5_fname = args.hdf5_fname
    list_selectAttribs = args.selectedAttribs

    if args.upsampling:
        path_to_GT = os.path.join(args.path2folder, "3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5")
        f_GT = h5py.File(path_to_GT, 'r')
        GT_dataMat = np.asarray(f_GT["dataMat"])
        GT_MEAN = np.asarray(f_GT["MEAN"])
        GT_STD = np.asarray(f_GT["STD"])
        GT_diminfo = np.asarray(f_GT["dim_info"])
        # GT_facemask = np.asarray(f_GT["facemask"])

    Ncompos = args.Ncompos
    with_faceMask = args.faceMask
    trackedMesh = args.trackedMesh

    if trackedMesh:
        datasetName = "trackedMesh"
        path_to_dataset=os.path.join(os.getcwd(), os.pardir, "samples", 'deformation_components', 'tracked_mesh')
    else:
        datasetName = "3dgs"
        path_to_dataset=os.path.join(os.getcwd(), os.pardir, "samples", 'deformation_components', 'trained_3dgs')
    assert os.path.exists(path_to_hdf5)

    path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples", "3dgs")


    if trackedMesh:
        NofFrames = 5
        tml = TrackedMeshLoader(path_to_dataset=os.path.join(os.getcwd(), os.pardir, os.pardir, "dataset", "multiface", "tracked_mesh"), ID=6795937, suffix='E0', mesh_loader="trimesh", num_samples_perExp=NofFrames)
        dataMat, _, _ = tml()
        MEAN = np.mean(dataMat, axis =0)
        STD = np.std(dataMat)
        dataMat = (dataMat - MEAN[None, :])/STD
        dataMat = dataMat.reshape(dataMat.shape[0], -1)
        list_keys = ["trackedMesh"]
        dataMats = [dataMat]
        numVerts = int(dataMat.shape[1]/3)
        print(numVerts)
        dim_info = [dataMat.shape[1]]
        list_attrName = list_keys
        mesh = trimesh.load(os.path.join(os.getcwd(), os.pardir, "samples","sample.obj"), force='mesh')
        tris = np.asarray(mesh.faces)
    else:
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
        print(f"0: xyz")
        print(dataMat.shape)
        list_attrName.append("xyz")
        dim_info.append(dataMat.shape[1])

        outputAttr_names=[]

        for i, key in enumerate(list_attribs):
            if key == "xyz":
                continue
            elif any(key in x for x in list_selectAttribs):
                print(f"{i+1}:{key}")
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
                dataMat = np.concatenate((dataMat, new_attributes), axis=1)
                print(new_attributes.shape)
        print(f"Final data mat shape: {dataMat.shape}")

        list_keys = [datasetName]
        dataMats = [dataMat]

        print("list of attribute in dataMat")
        print(list_attrName)
        print("list of dimension in dataMat for each attribute")
        print(dim_info)

        if numVerts == 5509:
            mesh = trimesh.load(os.path.join(path_to_sample, "sample_face.ply"), force = 'mesh')
            tris = np.asarray(mesh.faces)
        elif  numVerts == 21954:
            mesh = trimesh.load(os.path.join(path_to_sample, "sample_subd_face.ply"), force = 'mesh')
            tris = np.asarray(mesh.faces)
        elif numVerts == 87652:
            mesh = trimesh.load(os.path.join(path_to_sample, "sample_subd2_face.ply"), force = 'mesh')
            tris = np.asarray(mesh.faces)

    # print(dim_info)

    # loading facemask
    if with_faceMask:
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
    for key, dataMat in zip(list_keys, dataMats):
        print(f"-------------{key}-------------------")
        shape_dataMat = dataMat.shape
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

        # Ncompos = 50
        # # error calcuration w.r.t Ncompos samples
        # if with_faceMask:
        #     masked_PCA_dc = Gamma * facemask.bit_mask[None, :]
        #     # RMSE w.r.t masked normalized centralized X
        #     pca_RMSE = np.sqrt(((GTX - np.dot(Sigma[:, :Ncompos], (masked_PCA_dc[:Ncompos, :]).reshape(Ncompos, -1)))**2).mean())
        #     # sparsity
        #     pca_sparsity = np.sum(np.sqrt((masked_PCA_dc[:Ncompos, :]**2).sum(axis = 1)))
        #     pca_sparsity_level = np.mean(masked_PCA_dc == 0)
        # else:
        #     PCA_dc = Gamma
        #     Check_Reconstruction = (GTX - np.dot(Sigma, Gamma)).mean()
        #     print(Check_Reconstruction)
        #     # RMSE w.r.t masked normalized centralized X
        #     pca_RMSE = np.sqrt(((GTX - np.dot(Sigma[:, :Ncompos], (PCA_dc[:Ncompos, :]).reshape(Ncompos, -1)))**2).mean())
        #     # sparsity
        #     pca_sparsity = np.sum(np.sqrt((PCA_dc[:Ncompos, :]**2).sum(axis = 1)))
        #     pca_sparsity_level = np.mean(PCA_dc == 0)
        
        # print(f"PCA: RMSE_{pca_RMSE}, Sparsity_{pca_sparsity}, SparsityLevel_{pca_sparsity_level}")

        # # MBSPCA for Ncompos samples
        mb_sparsePCA = MiniBatchSparsePCA(n_components=Ncompos, verbose=False)
        est_MBSPCA = mb_sparsePCA.fit(GTX)

        C_mbspca = est_MBSPCA.components_.reshape(Ncompos, -1) #right hand side V
        if with_faceMask:
            W_mbspca = est_MBSPCA.transform(masked_GTX.reshape(masked_GTX.shape[0], -1)) #left hand side U
        else:
            W_mbspca = est_MBSPCA.transform(GTX) #left hand side U

        print(f"W shape: {W_mbspca.shape}")
        print(f"C shape: {C_mbspca.shape}")

        # if with_faceMask:
        #     masked_MBSPCA_dc = C_mbspca * facemask.bit_mask[None, :]
        #     mbspca_RMSE = np.sqrt(((GTX - np.dot(W_mbspca, (masked_MBSPCA_dc).reshape(masked_MBSPCA_dc.shape[0], -1)))**2).mean())
        #     mbspca_sparsity = np.sum(np.sqrt((masked_MBSPCA_dc**2).sum(axis = 2)))
        #     mbspca_sparsity_level = np.mean(masked_MBSPCA_dc==0)
        # else:
        #     mbspca_RMSE = np.sqrt(((GTX - np.dot(W_mbspca, (C_mbspca).reshape(C_mbspca.shape[0], -1)))**2).mean())
        #     mbspca_sparsity = np.sum(np.sqrt((C_mbspca**2).sum(axis = 2)))
        #     mbspca_sparsity_level = np.mean(C_mbspca==0)

        # print(f"MBSPCA: RMSE_{mbspca_RMSE}, Sparsity_{mbspca_sparsity}, SparsityLevel_{mbspca_sparsity_level}")
        if args.upsampling:
            DCs.update({key: {"dataMat": GT_dataMat, "faceMask": None, "MEAN": GT_MEAN, "STD": GT_STD, "pca_W": Sigma, "pca_C": Gamma, "mbspca_W": W_mbspca, "mbspca_C": C_mbspca, "tris": tris, "dim_info": GT_diminfo}})
        else:
            if with_faceMask:
                DCs.update({key: {"dataMat": dataMat, "faceMask": facemask.bit_mask, "MEAN": MEAN, "STD": STD, "pca_W": Sigma, "pca_C": Gamma, "mbspca_W": W_mbspca, "mbspca_C": C_mbspca, "tris": tris, "dim_info": dim_info}})
            else:
                DCs.update({key: {"dataMat": dataMat, "faceMask": None, "MEAN": MEAN, "STD": STD, "pca_W": Sigma, "pca_C": Gamma, "mbspca_W": W_mbspca, "mbspca_C": C_mbspca, "tris": tris, "dim_info": dim_info}})

    # print(PCAs)
    # print(MBSPCAs)

    hdf5_saveName = datasetName+"_"+str(numVerts)+"_ALL_5perExp_trimesh_dcs.hdf5"

    print(f"Saving PCA and MBSCPA at {os.path.join(path_to_dataset,hdf5_saveName)}")
    f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName), 'w')

    print(DCs.keys())

    dc_key = datasetName
    dset = f.create_dataset(name = "dim_info", data=np.asarray(DCs[dc_key]["dim_info"]))
    dset.attrs["list_attrName"] = str(list_attrName)
    dset1 = f.create_dataset(name = "dataMat", data=DCs[dc_key]["dataMat"])
    if with_faceMask:
        dset2 = f.create_dataset(name = "faceMask", data=DCs[dc_key]["faceMask"])
    dset3 = f.create_dataset(name = "MEAN", data = DCs[dc_key]["MEAN"])
    dset4 = f.create_dataset(name = "STD", data = DCs[dc_key]["STD"])
    dset5 = f.create_dataset(name = "pca_W", data = DCs[dc_key]["pca_W"])
    dset6 = f.create_dataset(name = "pca_C", data = DCs[dc_key]["pca_C"])
    dset7 = f.create_dataset(name = "mbspca_W", data = DCs[dc_key]["mbspca_W"])
    dset8 = f.create_dataset(name = "mbspca_C", data = DCs[dc_key]["mbspca_C"])
    dset9 = f.create_dataset(name = "tris", data = DCs[dc_key]["tris"])

    # check if they are actually stored in hdf5
    out_f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName))
    print(list(out_f.keys()))
    out_set = out_f["dim_info"]
    print(out_f["pca_W"].shape)
    print(out_f["pca_C"].shape)
    print(out_f["mbspca_W"].shape)
    print(out_f["mbspca_C"].shape)
    print(out_f["dim_info"])

'''
Execution sample
    conda activate 3dsrf
    python PCA_MBSPCA_3DGS.py --path2folder="../samples/deformation_components/trained_3dgs" --hdf5_fname="f336a291-bnotALLcam_datamat_5509.hdf5" --upsampling
'''

if __name__ == "__main__":
    parser = ArgumentParser(description="PCA and MBSPCA")
    parser.add_argument('--path2folder', type=str, default=None)
    parser.add_argument('--hdf5_fname', type = str, default=None)
    parser.add_argument('--selectedAttribs', type=list, default=["xyz", "f_dc", "rotation", "scale"])
    parser.add_argument('--Ncompos', type = int, default = 330)
    parser.add_argument('--faceMask', action = 'store_true', default = False)
    parser.add_argument('--trackedMesh', action = 'store_true', default = False)
    parser.add_argument('--upsampling', action= 'store_true', default = False)
    args = parser.parse_args(sys.argv[1:])

    PCA_MBSPCA_selectedAttribsAtOnce(args=args)