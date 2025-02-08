import os, sys
import h5py
import numpy as np
import trimesh
import trame
from scipy import sparse
from scipy import linalg
from scipy.sparse.linalg import splu
from scipy.optimize import curve_fit

sys.path.append(os.pardir)

from utils.pickel_io import load_from_memory
from utils.Blendshape import DeformationComponents
from utils.Geodesic_dist import compute_topological_laplacian
from utils.vis_tools import VisPointsAttributes
from utils.OBJ_helper import OBJ
from utils.Blendshape import FaceMask
from utils.Blendshape import ZeroMeanDefMatrix
from utils.Geodesic_dist import compute_topological_laplacian
from utils.vis_tools import VisPointsAttributes
from utils.pickel_io import dump_pckl, load_from_memory
from utils.Geodesic_dist import GeodesicDistHeatMethod, GeodesicDistSimple, compute_support_map, compute_support_map_gauss
from utils.converter import vector2MatNx3
from utils.common_utils import project_weight, proxy_l1l2
from argparse import ArgumentParser, Namespace

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

path_to_sample = os.path.join(os.getcwd(), os.pardir, "samples", "3dgs")

def gaussDist(x, a, mu, sigma):
    return a * np.exp(-(x-mu)**2/(2*sigma**2))

def attribute_subd2(attribute, sample_mesh):
    """
    sample_mesh: should be mesh in trimesh data structure (5509 verts)
    attribute: attribute of each vertices (attribute should be in shape [5509, #attribs])
    """
    subd1_vertices, subd1_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((sample_mesh.vertices, attribute)),
        faces = sample_mesh.faces
        )

    subd1_verts = subd1_vertices[:, :3]
    subd1_phiheat = subd1_vertices[:, 3:]

    subd2_vertices, subd2_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((subd1_verts, subd1_phiheat)),
        faces = subd1_faces
    )

    final_phi = subd2_vertices[:, 3:]
    final_verts = subd2_vertices[:, :3]
    return final_phi, final_verts

def SLDC_trained3dgs(args):

    path_to_dataset = args.path2folder
    pca_hdf5 = args.hdf5_fname
    dataset = args.representation_type
    sigma = args.sigma
    # get identical data matrix from PCA and MBSPCA

    # tracked mesh
    if dataset == "tracked_mesh":
        # pca_hdf5 = "tracked_mesh_5perExp_trimeshPCA_dcs.hdf5"
        pca_f = h5py.File(os.path.join(path_to_dataset, pca_hdf5), 'r')
    elif dataset == "trained_3dgs":
        # pca_hdf5 = "3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5"
        pca_f = h5py.File(os.path.join(path_to_dataset, pca_hdf5), 'r')

    dataMat = np.asarray(pca_f['dataMat'])
    dataMat = dataMat.reshape(dataMat.shape[0], -1, 3)
    
    path_to_facemaskPKL = os.path.join(os.getcwd(), os.pardir, "samples", "memory")

    pickel_fname = "FaceMask_sample_subd2_face_22012024_12_51_trimesh.pkl"
    assert os.path.exists(os.path.join(path_to_facemaskPKL,pickel_fname))
    print(f"loading facemask from {path_to_facemaskPKL}")
    _facemask = load_from_memory(path_to_memory=path_to_facemaskPKL, pickle_fname=pickel_fname)

    # facemask = np.asarray(pca_f["faceMask"], dtype = int)
    # facemask = facemask.reshape((-1, 3))
    facemask = _facemask.bit_mask
    print("facemask", facemask.shape)

    # MEAN: original vertex list of neutral face mesh
    # get the first frame of neutral face expression in the trained 3dgs
    MEAN = dataMat[0]

    Gaussian_MEAN = np.asarray(pca_f['MEAN'])
    Gaussian_STD = np.asarray(pca_f["STD"])

    # tris: triangle list (index tuples for triangle mesh)
    tris = np.asarray(pca_f["tris"])

    # get the number of verts
    Nverts = int(MEAN.shape[0])
    N_cent_X = dataMat
    mesh_loader = "trimesh"

    print(f"shape of data matrix: {N_cent_X.shape}")
    print(f"shape of triangle list: {tris.shape}")
    print(f"Number of vertices: {Nverts}")
    print(f"Mesh loader: {mesh_loader}")

    # reference verts and tris from the sample object file
    mesh = trimesh.load(os.path.join(path_to_sample, "sample_face.ply"), force='mesh')
    list_vertices = mesh.vertices
    list_triangles = mesh.faces
    verts = np.asarray(list_vertices)
    tris = np.asarray(list_triangles)

    MEAN_5509 = verts
    TRIS_5509 = tris

    if tris.min() > 0:
        for triangle in tris:
            for i in range(3):
                # print(triangle)
                triangle[i] = triangle[i] - int(1)

    # down-samplingoptimize
    R = N_cent_X[:, :5509, :3].copy()
    dataMat_5509 = N_cent_X[:, :5509, :3].copy()

    # number of components
    Ncompos = 330

    # minimum/maximum geodesic distance for support region 
    srMinDist = 0.1
    srMaxDist = 0.6

    # number of iterations to run
    num_iters_max = 20

    # sparsity parameter (coeffient lambda for weight of L1 regularization term)
    sparse_lambda = args.sparse_lambda

    sample_5509_mesh = trimesh.load(os.path.join(path_to_sample, "sample_face.ply"), force='mesh')
    # pernalty parameter for ADMM (for multiplier)
    # Choice of ρ can greatly influence practical convergence of ADMM
    # TOO large: not enough emphasis on minimizing a f+z
    # TOO small: not enought emphasis on feasibility (Ax+Bz = c) 
    rho = args.rho

    # number of iteration of ADMM
    num_admm_iterations = 10

    # geodesic distance computation on the mean verts
    gdd = GeodesicDistHeatMethod(MEAN_5509, TRIS_5509)

    C = []
    W = []

    for k in range(Ncompos):
        # find the vertex explaining the most variance across the residual matrix R
        # take a norm of residual at each vertex
        # print(f"shape: {R.shape}, {facemask[None, :5509, :].shape}")
        # magnitude = (R**2).sum(axis = 2) #shape [FxN]
        masked_R = R * facemask[None, :5509, :]
        magnitude = (masked_R**2).sum(axis = 2) #shape [FxN]

        # vertex id with the most variance (residual)
        idx = np.argmax(magnitude.sum(axis = 0))

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
        if args.gauss:
            flippedSR = 1 - compute_support_map_gauss(phi=phi, mu=0.0, sigma=np.sqrt(sigma))
        else:
            flippedSR = 1 - compute_support_map(phi, srMinDist, srMaxDist)

        # Solve normal equation to get C_k
        # R: shape = [F, N, 3]
        # W_K: shape = [F, 1]
        # c_k: shape = [N, 3]
        # flippedSR: shape = [N, ]
        # W_k*C_k = flippedSR*R
        # C_k = (W_k^T*W_k)^{-1} W_k^T*flippedSR*R (least square solution)

        c_k = (np.tensordot(w_k, R, (0, 0)) * flippedSR[:, None])/ np.inner(w_k, w_k)

        C.append(c_k)
        W.append(w_k)

        # update residual
        R = R - np.outer(w_k, c_k).reshape(R.shape)

    C = np.array(C) #shape = [K, N, 3]
    print("C", C.shape)
    print("face mask", facemask[None, :5509, :].shape)
    C = C * facemask[None, :5509, :]
    W = np.array(W).T #shape = [F, K]

    original_error = (dataMat_5509**2).sum()

    # global optimization
    # F, N, _ = Nmasked_cent_X.shape
    F, N, _ = dataMat_5509.shape

    Lambda = np.empty((Ncompos, N)) # each row representing the scaler of l1 penalty depending on the locality
    U = np.zeros_like(C)
    print(U.shape)

    list_reconstruction_errors = []
    list_sparsity = []

    R = dataMat_5509.copy()

    for i in range(num_iters_max):
        # Update weights
        # fix weight matrix, optimize C (each row respectively: c_k)
        # R = R * facemask[None, :5509, :]
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
            # deformation_aware_supportMap = 1 - gaussDist(x = linear_int_x, a=params[0], mu = params[1], sigma = params[2])
            # find vertex with the biggest displacement in component and computer support map around it
            # take displacement vector norm at each vertex and find index with maximum of norm
            idx = (ck**2).sum(axis = 1).argmax()
            phi = gdd(idx)
            phi/=max(phi)
            if args.gauss:
                support_map = compute_support_map_gauss(phi=phi, mu=0.0, sigma=np.sqrt(sigma))
            else:
                support_map = compute_support_map(phi, srMinDist, srMaxDist)
            
            # flatten_ck = ck.ravel()
            # len_x = ck.shape[1]
            # linear_int_x = np.linspace(0, len_x, 1, dtype=int)
            # params, pcov = curve_fit(gaussDist, linear_int_x, flatten_ck)
            # final_supportMap = 0.5 * deformation_aware_supportMap + 0.5 * support_map
            # update L1 regularization strength according to this support map
            Lambda[k] = sparse_lambda * support_map
        
        # TODO
        # Inf or NaN check in W and C

        # update components
        Z = C.copy() # this is dual variable

        # optimize matrix C fixing W
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        # G[np.isfinite(G) == False] = 0
        # c = np.dot(W.T, Nmasked_cent_X.reshape(Nmasked_cent_X.shape[0], -1)) #Nmasked_cent_X.reshaped into [F, N*3]  
        c = np.dot(W.T, dataMat_5509.reshape(dataMat_5509.shape[0], -1)) #Nmasked_cent_X.reshaped into [F, N*3]  
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
        R = dataMat_5509 - np.tensordot(W, C, (1, 0)) # residual
        if (i == 0):
            initial_sparsity = np.sum(np.sqrt((C**2).sum(axis = 2))) # L1 reguralization term 
            initial_reconst_error = np.sqrt(((dataMat_5509.reshape(dataMat_5509.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())

        # sparsity = np.sum(np.sqrt(((C*facemask[None, :])**2).sum(axis = 2))) # L1/L2 reguralization term
        sparsity = np.mean(C==0)
        # reconstruction error: root mean squared error * 1000 for convenience
        # reconst_error = np.sqrt(((X.reshape(X.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())/1e3
        reconst_error = np.sqrt(((dataMat_5509.reshape(dataMat_5509.shape[0], -1) - np.dot(W, C.reshape(C.shape[0], -1)))**2).mean())
        # print(f"Reconstruction error: {(reconst_error/initial_reconst_error)}")
        print(f"Reconstruction error: {(reconst_error)}")
        list_reconstruction_errors.append(reconst_error)
        list_sparsity.append(sparsity)
        print(f"Sparsity: {sparsity}")
        # e = ((reconst_error/initial_reconst_error)) + sparsity/initial_sparsity
        e = ((reconst_error)) + sparsity

        # convergence check
        print("iteration %03d, E=%f" % (i, e))
    
    # up-sampling
    temp_C = C.copy()
    print(temp_C.shape)
    reshaped_C = temp_C[0]
    for i in range(1, temp_C.shape[0]):
        reshaped_C = np.concatenate((reshaped_C, temp_C[i]), axis = 1)
    print(reshaped_C.shape)
    C_subd2, _ = attribute_subd2(attribute=reshaped_C, sample_mesh = sample_5509_mesh)
    final_C = None
    final_C = C_subd2[:, :3].reshape(1, 87652, 3)
    print(final_C.shape)
    for i in range (1, temp_C.shape[0]):
        final_C = np.concatenate((final_C, C_subd2[:, 3*i:3*(i+1)].reshape(1, 87652, 3)), axis = 0)
    # final_C = final_C * facemask
    if args.gauss:
        hdf5_saveName = "gauss_3dgs_"+str(Nverts)+"_xyz_SLDC_5perExp_trimesh_dcs_"+str(sigma)+"_"+str(sparse_lambda)+"_"+str(rho)+".hdf5"
    else:
        hdf5_saveName = "3dgs_"+str(Nverts)+"_xyz_SLDC_5perExp_trimesh_dcs"+"_"+str(sparse_lambda)+".hdf5"
    datasetName = "3dgs"

    f = h5py.File(os.path.join(path_to_dataset, hdf5_saveName), 'w')
    # verts 5509
    # [xyz, f_dc, rotation, scale]
    # [16527, 16527, 22036, 16527]
    # verts 87652
    # [xyz, f_dc, rotation, scale]
    # [262956, 262956, 350608, 262956]
    dc_key = datasetName
    list_attrName = ["xyz", "f_dc", "rotation", "scale"]
    dim_info = [262956, 262956, 350608, 262956]
    dset = f.create_dataset(name = "dim_info", data=np.asarray(dim_info))
    dset.attrs["list_attrName"] = str(list_attrName)
    dset1 = f.create_dataset(name = "dataMat", data=dataMat)
    dset2 = f.create_dataset(name = "faceMask", data=facemask)
    dset3 = f.create_dataset(name = "MEAN", data = Gaussian_MEAN)
    dset4 = f.create_dataset(name = "STD", data = Gaussian_STD)
    dset5 = f.create_dataset(name = "sldc_W", data = W)
    dset6 = f.create_dataset(name = "sldc_C", data = final_C)
    dset9 = f.create_dataset(name = "tris", data = tris)


    '''
    Execution sample
        conda activate 3dsrf
        python SLDC_trained3dgs.py --path2folder="../samples/deformation_components/trained_3dgs" --hdf5_fname="3dgs_87652_xyz_5perExp_trimesh_dcs.hdf5 --gauss"
    '''

if __name__ == "__main__":

    parser = ArgumentParser(description="SLDC")
    parser.add_argument('--path2folder', type = str, default="")
    parser.add_argument('--hdf5_fname', type=str, default="3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5")
    parser.add_argument('--representation_type', type=str, default="trained_3dgs")
    parser.add_argument('--gauss', action='store_true', default=False)
    parser.add_argument('--sigma', type = float, default=0.3)
    parser.add_argument('--sparse_lambda', type = float, default=2.0)
    parser. add_argument('--rho', type = float, default=10.0)
    args = parser.parse_args(sys.argv[1:])

    if args.sigma:
        print("sigma: ", args.sigma)
        print("sparse_lambda", args.sparse_lambda)
        print("rho:", args.rho)

    SLDC_trained3dgs(args=args)