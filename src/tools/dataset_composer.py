"""
create a folder of each time_stampt
- For 3D Gaussian splatting, we need to convert data structure into following to collect images which are taken at the same time (time_stamp)

     1.   `makedir ./dataset/multi_views`
     2.   `create ./dataset/eachTimeStamptImages/ID/Exp/TimeStamp/Cam for each timestamp`
     3.   `load retrieve image from each folder and save into the ./ID/Exp/TimeStamp/Cam`
"""

import os, sys
import shutil 
import trimesh
from PIL import Image
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

path_to_src = os.pardir
sys.path.append(path_to_src)
from utils.Dataset_handler import Filehandler
from utils.OBJ_helper import OBJ
from utils.metadata_loader import load_KRT, load_RT
from utils.PLY_helper import tex2vertsColor, save_ply, gammaCorrect

color_corr = lambda x: (255 * gammaCorrect(x / 255.0, dim=2)).clip(0, 255)

# path to targetDir
path_to_targetDir = os.path.join(os.getcwd(), os.pardir, os.pardir, 'dataset')
# path to dir to store images
path_to_MultiViews = os.path.join(path_to_targetDir, 'multi_views')
# [TODO] path to image folder which is downloade from multi-face
path_to_imagesFolder = ""
ID = '6795937'

# path to dataset we will load from /mnt/hdd
assert os.path.exists(path_to_targetDir) == True
assert os.path.exists(path_to_MultiViews) == True

path_to_ID = os.path.join(path_to_MultiViews, ID)
if not os.path.exists(path_to_MultiViews):
    os.mkdir(path_to_MultiViews)
if not os.path.exists(path_to_ID):
    os.mkdir(path_to_ID)

assert os.path.exists(path_to_imagesFolder) == True

# since we download tracked mesh in /home, we will use tracked mesh from there
path_to_trackedmesh = os.path.join(path_to_targetDir, 'multiface', 'tracked_mesh')
assert os.path.exists(path_to_trackedmesh) == True

dict_images = {}
list_dirNames, list_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=path_to_imagesFolder, prefix='E0')

# from the first frame
NofFrames = 5

# print(list_dirNames)
# print(list_dirPaths)
counter = 0
print("Loading and copying only necesarry dataset to ->", path_to_ID)
for dirName, dirPath in zip(list_dirNames, list_dirPaths):
    path_to_expImageFolder = os.path.join(path_to_ID, dirName)
    expName = dirName
    print("-"*10)
    print(expName)
    if not os.path.exists(path_to_expImageFolder):
        os.mkdir(path_to_expImageFolder)
    
    list_CamDirNames, list_CamDirPaths = Filehandler.dirwalker_InFolder(path_to_folder=dirPath, prefix='400')
    print("the number of Cameras: ", len(list_CamDirNames))
    for camDirName, camDirPath in zip(list_CamDirNames, list_CamDirPaths):
        list_imageNames, list_imagePaths = Filehandler.fileWalker_InDirectory(path_to_directory=camDirPath, ext='.png')
        print("Camera ID: ", camDirName)
        print(f"First {NofFrames} frames: {list_imageNames[:NofFrames]}")
        list_imageNames = list_imageNames[:NofFrames]
        list_imagePaths = list_imagePaths[:NofFrames]
        # print(list_imageNames)
        # print(list_imagePaths)
        for imageN, imagePath in zip(list_imageNames, list_imagePaths):
            time_stamp = imageN[:6]
            path_to_tsDir = os.path.join(path_to_expImageFolder, time_stamp)
            if not os.path.exists(path_to_tsDir):
                os.mkdir(path_to_tsDir)
            image_save_name = camDirName+'.png'
            path_to_mesh = os.path.join(path_to_trackedmesh, expName, str(time_stamp)+'.obj')
            path_to_transTXT = os.path.join(path_to_trackedmesh, expName, str(time_stamp)+'_transform.txt')
            # print(path_to_mesh)
            assert os.path.exists(path_to_mesh) == True
            # print(time_stamp)
            # print(os.path.join(path_to_tsDir, image_save_name))
            if not os.path.exists(os.path.join(path_to_tsDir, image_save_name)):
                shutil.copyfile(src=imagePath, dst=os.path.join(path_to_tsDir, image_save_name))

            if not os.path.exists(os.path.join(path_to_tsDir, str(time_stamp) + '.obj')):
                shutil.copyfile(src=path_to_mesh, dst = os.path.join(path_to_tsDir, str(time_stamp) + '.obj'))

            if not os.path.exists(os.path.join(path_to_tsDir, str(time_stamp)+'_transform.txt')):
                shutil.copyfile(src=path_to_transTXT, dst = os.path.join(path_to_tsDir, str(time_stamp) + '_transform.txt'))
            else:
                continue
    counter = counter + 1

# Start to creat ply and save them in the appropriate directory
print("Generating PLY and save in the corresponding timestamp dir")
path_to_averageTex = os.path.join(path_to_targetDir, 'multiface', 'meta_data', 'tex_mean.png')
assert os.path.exists(path_to_averageTex) == True
np_tex = np.array(Image.open(path_to_averageTex))
# Correct the skin tone by using color corrector
np_tex = color_corr(np_tex)
print("average texture (shape):", np_tex.shape)

tex = Image.fromarray(np.uint8(np_tex))
# plt.imshow(tex)

meshes = []
mesh_paths = []
trimeshes = []
list_expNames, list_expPaths =  Filehandler.dirwalker_InFolder(path_to_folder=path_to_ID, prefix='E0')
for expName, expPath in zip(list_expNames, list_expPaths):
    print(expName)
    print("path: ", expPath)
    counter = 0
    list_tsNames, list_tsPaths = Filehandler.dirwalker_InFolder(path_to_folder=expPath, prefix='0')
    print(list_tsNames)
    assert len(list_tsPaths) == NofFrames
    for time_stamp, ts_Path in zip(list_tsNames, list_tsPaths):
        print(time_stamp)
        mesh_path = os.path.join(ts_Path, time_stamp+'.obj')
        assert os.path.exists(mesh_path)

        mesh = trimesh.load(file_obj=mesh_path, process='mesh')
        head_pose = load_RT(os.path.join(ts_Path, str(time_stamp)+"_transform.txt"))
        vertices  = np.asarray(mesh.vertices)

        vertex_normals = np.asarray(mesh.vertex_normals)
        print("the number of verts: ", vertices.shape)
        # print(vertices)
        faces = np.asarray(mesh.faces)
        print("the number of faces: ", faces.shape)
        # print(faces)
        pv_mesh = pv.wrap(mesh)
        texture_coords = np.asarray(pv_mesh["Texture Coordinates"])
        tex_lenu, tex_lenv = np_tex.shape[:2]
        vertex_colors = tex2vertsColor(texture_coordinate=texture_coords, np_tex=np_tex, tex_lenu=tex_lenu, tex_lenv = tex_lenv)
        # print(vertex_colors.shape)
        # print(vertex_colors)
        # meshes.append(mesh)

        save_ply(os.path.join(ts_Path, time_stamp +'.ply'), vertices=vertices, faces=faces, vertex_normals=vertex_normals, vertex_colors=vertex_colors, only_points=True)

        # subdivision (every face subdivided into 4 faces)
        subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=vertices, faces = faces, face_index=None, vertex_attributes={"normal": vertex_normals, "color":vertex_colors}, return_index=True)
        subd_vertex_normals = subd_dict["normal"]
        subd_vertex_colors = subd_dict["color"]
        save_ply(os.path.join(ts_Path, time_stamp +'_subd.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=True)
        
        # sub-subdivision (every face subdivided into 4 faces)
        subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors}, return_index=True)
        subd_vertex_normals = subd_dict["normal"]
        subd_vertex_colors = subd_dict["color"]
        save_ply(os.path.join(ts_Path, time_stamp +'_subd2.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=True)


print("Done")