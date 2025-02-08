import numpy as np
def gammaCorrect(img, dim=-1):

    if dim == -1:
        dim = len(img.shape) - 1 
    assert(img.shape[dim] == 3)
    gamma, black, color_scale = 2.0,  3.0 / 255.0, [1.4, 1.1, 1.6]

    if dim == -1:
        dim = len(img.shape) - 1

    scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
    img = img * scale / 1.1
    return np.clip((((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2, )


def tex2vertsColor(texture_coordinate, np_tex, tex_lenu, tex_lenv):
    vertex_colors = []
    for i, tex_coord in enumerate(texture_coordinate):
        # print(tex_coord)
        x, y = np.round(((1-tex_coord[1])*(tex_lenu-1))%tex_lenu).astype(np.int64), np.round(((tex_coord[0])*(tex_lenv-1))%tex_lenv).astype(np.int64)
        # print(x, y)
        color = np_tex[x, y, :]
        vertex_colors.append(color)
        # print(color)
    return np.asarray(vertex_colors)

def save_ply(f_name, vertices, faces, vertex_normals, vertex_colors, only_points = False):
    """
    Arguments
        vertices: np.array (N, 3)
        faces: np.array (F, 3)
        vertex_colors: np.array (N, 3)

    Returns
        ply file

    """
    
    header = "ply\n" + "format ascii 1.0\n"
    with open(f_name, 'w') as f:
        # write header
        f.write(header)
        # number of vertex
        num_verts = vertices.shape[0]
        f.write("element vertex "+str(num_verts)+"\n")
        # write vertex property
        verts_property = "property float x\n" + "property float y\n" + "property float z\n" + "property float nx\n" + "property float ny\n" + "property float nz\n" + "property uchar red\n" + "property uchar green\n" + "property uchar blue\n"
        f.write(verts_property)

        # number of faces
        if not only_points:
            num_faces = faces.shape[0]
            f.write("element face "+str(num_faces) + "\n")
            # write face property
            faces_property = "property list uchar int vertex_index\n"
            f.write(faces_property)

        f.write("end_header\n")

        # write vertex location
        for coord, normal, color in zip(vertices, vertex_normals,vertex_colors):
            v_line = ""
            v_line = str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + " " + str(normal[0]) +" "+ str(normal[1]) + " " + str(normal[2]) + " " + str(np.round(color[0]).astype(np.int64)) + " " + str(np.round(color[1]).astype(np.int64)) + " " + str(np.round(color[2]).astype(np.int64)) + "\n"
            f.write(v_line)

        if not only_points:
        # write triangle list
            for index3 in faces:
                t_line = ""
                t_line = "3 " + str(index3[0]) + " " + str(index3[1]) + " " + str(index3[2]) + "\n"
                f.write(t_line)
    f.close()