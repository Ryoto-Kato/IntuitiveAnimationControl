import os, sys
import numpy as np
import trimesh
import h5py
from scipy import sparse
from scipy import linalg
from scipy.sparse.linalg import splu

sys.path.append(os.pardir)
from utils.pickel_io import load_from_memory
from utils.Blendshape import ZeroMeanDefMatrix
from utils.Geodesic_dist import compute_topological_laplacian
from utils.vis_tools import VisPointsAttributes
from utils.vis_tools import VisPointsAttributes
from utils.OBJ_helper import OBJ
from utils.Blendshape import FaceMask
from utils.Blendshape import ZeroMeanDefMatrix
from utils.Geodesic_dist import compute_topological_laplacian
from utils.vis_tools import VisPointsAttributes
from utils.pickel_io import dump_pckl, load_from_memory
from utils.Geodesic_dist import GeodesicDistHeatMethod, GeodesicDistSimple, compute_support_map
from utils.converter import vector2MatNx3
from utils.common_utils import project_weight, proxy_l1l2

from scipy import sparse
from scipy import linalg
from scipy.sparse.linalg import splu

# use efficient implementation of sparse Cholesky factorization.
from sksparse.cholmod import cholesky_AAt, cholesky


from datetime import datetime
# get current date and year
now = datetime.now()

date = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")
print(date)
time = now.strftime("%H_%M")
print("time:", time)

path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples")

def SLDC_trackedMesh():
    # dataset = "trained_3dgs"
    dataset = "tracked_mesh"

    if dataset == "tracked_mesh":
        path_to_dataset = os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", dataset)
    elif dataset == "trained_3dgs":
        path_to_dataset = os.path.join(os.getcwd(), os.pardir, "samples", "deformation_components", dataset)

    # get identical data matrix from PCA and MBSPCA

    # tracked mesh
    if dataset == "tracked_mesh":
        pca_hdf5 = "trackedMesh_5509_ALL_5perExp_trimesh_dcs.hdf5"
        pca_f = h5py.File(os.path.join(path_to_dataset, pca_hdf5), 'r')
    elif dataset == "trained_3dgs":
        pca_hdf5 = "3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs_usedForSLDC.hdf5"
        pca_f = h5py.File(os.path.join(path_to_dataset, pca_hdf5), 'r')

    dataMat = np.asarray(pca_f['dataMat'])
    dataMat = dataMat.reshape((dataMat.shape[0], -1, 3))
    facemask = np.asarray(pca_f["faceMask"], dtype=int)
    facemask = facemask.reshape((-1, 3))

    # MEAN: original vertex list of neutral face mesh
    # get the first frame of neutral face expression in the trained 3dgs
    MEAN = np.asarray(pca_f["MEAN"])

    STD = np.asarray(pca_f["STD"])

    # tris: triangle list (index tuples for triangle mesh)
    tris = np.asarray(pca_f["tris"])

    # get the number of verts
    Nverts = int(MEAN.shape[0])

    mesh_loader = "trimesh"

    print(f"shape of data matrix: {dataMat.shape}")
    print(f"shape of mean mesh vertex array: {MEAN.shape}")
    print(f"shape of triangle list: {tris.shape}")
    print(f"std of data matrix: {STD}")
    print(f"Number of vertices: {Nverts}")
    print(f"Mesh loader: {mesh_loader}")

    mesh = trimesh.load(os.path.join(path_to_sample, "sample.obj"), force='mesh')
    list_vertices = mesh.vertices
    list_triangles = mesh.faces

    verts = np.asarray(list_vertices)
    tris = np.asarray(list_triangles)

    if tris.min() > 0:
        for triangle in tris:
            for i in range(3):
                # print(triangle)
                triangle[i] = triangle[i] - int(1)
    print(dataMat)
    N_cent_X = dataMat
    # print(np.isfinite(N_cent_X).all())
    R = N_cent_X.copy()

    # number of components
    Ncompos = 50

    # minimum/maximum geodesic distance for support region 
    srMinDist = 0.01
    srMaxDist = 0.35

    # number of iterations to run
    num_iters_max = 50

    # sparsity parameter (coeffient lambda for weight of L1 regularization term)
    sparse_lambda = 2

    # pernalty parameter for ADMM (for multiplier)
    # Choice of ρ can greatly influence practical convergence of ADMM
    # TOO large: not enough emphasis on minimizing a f+z
    # TOO small: not enought emphasis on feasibility (Ax+Bz = c) 
    rho = 10.0

    # number of iteration of ADMM
    num_admm_iterations = 20

    # geodesic distance computation on the mean verts
    gdd = GeodesicDistHeatMethod(MEAN, tris)

    C = []
    W = []

    for k in range(Ncompos):
        # find the vertex explaining the most variance across the residual matrix R
        # take a norm of residual at each vertex
        # print(f"R shape: {R.shape}")
        masked_R = R * facemask[None, :]
        # print(f"masked_R shape: {R.shape}")

        # magnitude = (R**2).sum(axis = 2) #shape [FxN]
        magnitude = (masked_R**2).sum(axis = 2) #shape [FxN]
        print(magnitude.shape)
        # vertex id with the most variance (residual)
        idx = np.argmax((magnitude).sum(axis = 0))

        # Find linear component explaining the motion of this vertex
        # R: shape = [F, 3]
        _U, s, Vh = linalg.svd(R[:, idx, :].reshape(R.shape[0], -1).T, full_matrices=False)
        
        # reconstruct column of matrix W at K-th column using most variant direction
        w_k = s[0] * Vh[0, :]

        # invert weight according to their projection onto the constraint set
        # This prevent problems from having negative weights
        wk_proj = project_weight(w_k)
        wk_proj_negative = project_weight(-1*w_k)

        # W_k will be replaced by the larger variance direction (+ or -)
        if(linalg.norm(wk_proj) > linalg.norm(wk_proj_negative)):
            w_k = wk_proj
        else:
            w_k = wk_proj_negative

        # flipped support region
        phi = gdd(idx)
        phi/=max(phi)
        flippedSR = 1 - compute_support_map(phi, srMinDist, srMaxDist)

        # Solve normal equation to get C_k
        # R: shape = [F, N, 3]
        # W_K: shape = [F, 1]
        # c_k: shape = [N, 3]
        # flippedSR: shape = [N, ]
        # W_k*C_k = flippedSR*R
        # C_k = (W_k^T*W_k)^{-1} W_k^T*flippedSR*R

        c_k = (np.tensordot(w_k, R, (0, 0)) * flippedSR[:, None])/ np.inner(w_k, w_k)

        C.append(c_k)
        W.append(w_k)

        # update residual
        R = R - np.outer(w_k, c_k).reshape(R.shape)

    C = np.array(C) #shape = [K, N, 3]
    W = np.array(W).T #shape = [F, K]

    original_error = (R**2).sum()

    # global optimization
    # F, N, _ = Nmasked_cent_X.shape
    F, N, _ = R.shape

    Lambda = np.empty((Ncompos, N)) # each row representing the scaler of l1 penalty depending on the locality
    U = np.zeros_like(C)
    print(U.shape)

    list_reconstruction_errors = []
    list_sparsity = []

    for i in range(num_iters_max):
        # Update weights
        # fix weight matrix, optimize C (each row respectively: c_k)
        Rflat = R.reshape(F, N*3) #flattened residual, shape = [F, N*3]
        for k in range(C.shape[0]): # for c_k (kth row)
            c_k = C[k].ravel() #flatten into [1, N*3]
            ck_norm = np.inner(c_k, c_k)
            if ck_norm <= 1e-8: # if the component does not represent any deformation component
                W[:, k] = 0
                continue # to prevent dividing by 0
            
            #block coordinate descent update
            # get updated W[:,k]'
            Rflat += np.outer(W[:, k], c_k) 
            opt = np.dot(Rflat, c_k) / ck_norm 

            #project W onto the desired space from constraints
            W[:, k] = project_weight(opt)
            Rflat -= np.outer(W[:, k], c_k)

        # precomputing lambda for each component k (Regularization term)
        # spatially varying regularization strength (to encode locality)
        for k in range(Ncompos):
            ck = C[k] #not flatten
            # find vertex with the biggest displacement in component and computer support map around it
            # take displacement vector norm at each vertex and find index with maximum of norm
            idx = (ck**2).sum(axis = 1).argmax()
            phi = gdd(idx)
            phi/=max(phi)
            support_map = compute_support_map(phi, srMinDist, srMaxDist)

            # update L1 regularization strength according to this support map
            Lambda[k] = sparse_lambda * support_map
        
        # TODO
        # Inf or NaN check in W and C

        # update components
        Z = C.copy() # this is dual variable

        # optimize matrix C fixing W
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        # c = np.dot(W.T, Nmasked_cent_X.reshape(Nmasked_cent_X.shape[0], -1)) #Nmasked_cent_X.reshaped into [F, N*3]  
        c = np.dot(W.T, R.reshape(R.shape[0], -1)) #Nmasked_cent_X.reshaped into [F, N*3]  
        # compute inverse part
        # scipy
        solve_prefactored = linalg.cho_factor(G + rho * np.eye(G.shape[0]))
        # sksparse.cholmod
        # sparse_csc_c = sparse.csc_matrix(G + rho * np.eye(G.shape[0]))
        # solve_prefactored = cholesky(sparse_csc_c)

        # ADMM iterations
        # TODO
        #    - check cho_factor and cho_solve from scipy
        #    - create function for proxy of update of l1/l2 reguralization term
        # old_U = U.reshape(U.shape[0], -1)
        for admm_it in range(num_admm_iterations):
            # temp_U = U.reshape(U.shape[0], -1)
            # for i in range(temp_U.shape[0]):
            #     for j in range(temp_U.shape[1]):
            #         if not np.isfinite(temp_U[i][j]):
            #             if old_U[i][j] != 0.0:
            #                 print(f"{i}, {j}: {old_U[i][j]}")
            rhs = c + rho * (Z.reshape(c.shape) - U.reshape(c.shape))
            # rhs[np.isfinite(rhs)==False] = 0
            C = linalg.cho_solve(solve_prefactored, rhs).reshape(C.shape)
            # sparse_csc_rhs = sparse.csc_matrix(c + rho * (Z.reshape(c.shape) - U.reshape(c.shape)))
            # sparse_csc_lhs = solve_prefactored(sparse_csc_rhs)
            # C = sparse_csc_lhs.toarray().reshape(C.shape)
            Z = proxy_l1l2(Lambda, C+U, 1.0/rho)
            # old_U= U.reshape(U.shape[0], -1)
            U = U + C - Z

        # set updated components to dual Z
        C = Z

        # evaluate objective function
        # R = Nmasked_cent_X - np.tensordot(W, C, (1, 0)) # residual
        R = R - np.tensordot(W, C, (1, 0)) # residual
        if (i == 0):
            initial_sparsity = np.sum(np.sqrt((C**2).sum(axis = 2))) # L1 reguralization term 
            initial_reconst_error = np.sqrt(((dataMat.reshape(dataMat.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())

        # sparsity = np.sum(np.sqrt(((C*facemask.bit_mask[None, :])**2).sum(axis = 2))) # L1/L2 reguralization term
        sparsity = np.mean(C==0)
        # reconstruction error: root mean squared error * 1000 for convenience
        # reconst_error = np.sqrt(((X.reshape(X.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())/1e3
        reconst_error = np.sqrt(((dataMat.reshape(dataMat.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())
        # print(f"Reconstruction error: {(reconst_error/initial_reconst_error)}")
        print(f"Reconstruction error: {(reconst_error)}")
        list_reconstruction_errors.append(reconst_error)
        list_sparsity.append(sparsity)
        print(f"Sparsity: {sparsity}")
        # e = ((reconst_error/initial_reconst_error)) + sparsity/initial_sparsity
        e = ((reconst_error)) + sparsity

        # convergence check
        print("iteration %03d, E=%f" % (i, e))


    hdf5_saveName = "trackedMesh_5509_SLDC_5perExp_trimesh_dcs.hdf5"
    datasetName = "trackedMesh"

    f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName), 'w')


    dc_key = datasetName
    list_attrName = ["xyz", "f_dc", "rotation", "scale"]
    dim_info = [262956, 262956, 350608, 262956]
    dset = f.create_dataset(name = "dim_info", data=np.asarray(dim_info))
    dset.attrs["list_attrName"] = str(list_attrName)
    dset1 = f.create_dataset(name = "dataMat", data=dataMat)
    dset2 = f.create_dataset(name = "faceMask", data=facemask)
    dset3 = f.create_dataset(name = "MEAN", data = MEAN)
    dset4 = f.create_dataset(name = "STD", data = STD)
    dset5 = f.create_dataset(name = "sldc_W", data = W)
    dset6 = f.create_dataset(name = "sldc_C", data = C)
    dset9 = f.create_dataset(name = "tris", data = tris)


if __name__ == "__main__":
    SLDC_trackedMesh()

