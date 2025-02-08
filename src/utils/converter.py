import numpy as np

def vector2MatNx3(vector, num_verts):
    verts_pos_list = []
    _list = []
    for i in range(num_verts):
        _list = [vector[i*3], vector[i*3+1], vector[i*3+2]]
        verts_pos_list.append(_list)
    return np.asarray(verts_pos_list)

def vector3D2Scaler(vector, num_verts, normalized=False):
    verts_scalar = []
    scale = []
    for i in range(num_verts):
        scale = np.sqrt(vector[i*3]**2 + vector[i*3+1]**2 + vector[i*3+2]**2)
        verts_scalar.append(scale)
    if normalized:
        verts_scalar = verts_scalar/np.max(verts_scalar)
    return np.asarray(verts_scalar)

def vector2MatFxNx3(vector, num_verts):
    verts_pos_list = []
    _list = []
    for i in range(num_verts):
        _list = [vector[i*3], vector[i*3+1], vector[i*3+2]]
        verts_pos_list.append(_list)
    return np.asarray(verts_pos_list)

def MatFxNx32MatF3N(in_mat, num_verts):
    out_mat = np.empty((in_mat.shape[0], in_mat.shape[1]*in_mat.shape[2]))
    for i in range(in_mat.shape[0]):
        out_mat[i] = in_mat[i].ravel()
    return out_mat

