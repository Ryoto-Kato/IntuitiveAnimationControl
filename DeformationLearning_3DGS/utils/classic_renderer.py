import numpy as np
import sys, os
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# backend of off-screen rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from argparse import ArgumentParser, Namespace

path_to_3WI = os.path.join(os.getcwd(), os.pardir, os.pardir, "3DSSL-WS23_IntuitiveAnimation", "")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.Dataset_handler import Filehandler
from utils.metadata_loader import load_KRT, load_RT
from utils.PLY_helper import tex2vertsColor, gammaCorrect

parser = ArgumentParser(description="Classic rendering")
parser.add_argument('--ID', type=str, default = "6795937")
parser.add_argument('--expName', type = str, default = "E001_Neutral_Eyes_Open")
parser.add_argument('--timestamp', type=str, default = None)
parser.add_argument('--savefig', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

ID = args.ID
expName = args.expName
timestamp = args.timestamp
savefig = args.savefig

# python classic_render.py --expName=E001_Neutral_Eyes_Open  --savefig

path_to_dataset = os.path.join(path_to_3WI, "dataset")
path_to_multiface = os.path.join(path_to_dataset, "multiface")
path_to_metadata = os.path.join(path_to_multiface, "meta_data")

# Get the directory containing images at each time stamp in the expression folder
list_TimeStampDirNames, list_TimeStampDir_Paths = Filehandler.dirwalker_InFolder(path_to_folder = os.path.join(path_to_dataset, "COLMAP", ID, expName), prefix='0')

if timestamp == None:
    timestamp = list_TimeStampDirNames[0][:6]
print(f"Expression:{expName}, Timestamp:{timestamp}")

path_to_tsdir = list_TimeStampDir_Paths[0]

assert os.path.exists(path_to_tsdir) == True

ply_path = os.path.join(path_to_tsdir, timestamp+'.obj')
tracked_mesh = trimesh.load(file_obj = ply_path)

# for i, point in enumerate(tracked_mesh.vertices):
#     # print("before:",point)
#     tracked_mesh.vertices[i][1] *= -1
#     tracked_mesh.vertices[i][2] *= -1
#     # print("after:" ,tracked_mesh.vertices[i])

headpose = load_RT(os.path.join(path_to_tsdir, timestamp+"_transform.txt"))
f_KRT = "KRT"
meta_cameras = load_KRT(os.path.join(path_to_metadata, f_KRT))

# upscale factor (rendering_res = image_res * upscale factor)
uf = 1
img_dim = (1334*uf, 2048*uf) # width and height
for i, cam_name in enumerate(meta_cameras.keys()):
    if i == 0:
        camera_id = cam_name

        sys.stdout.write("Reading camera {}/{}\n".format(i+1, len(meta_cameras.keys())))
        sys.stdout.write("Name: {}\n".format(cam_name))
        sys.stdout.flush()

        #load extrinsic
        camera_pose = np.vstack((meta_cameras[camera_id]["extrin"], [0,0,0,1]))
        
        #convert camera extrinsic convention (openCV) to OPENGL
        print("before:", camera_pose)
        
        camera_pose[:3, 1:3] *= -1.0

        print("after:", camera_pose)
        
        # load intrinsicshape
        focal_x, focal_y = meta_cameras[camera_id]["intrin"][0, 0], meta_cameras[camera_id]["intrin"][1, 1]
        cx, cy = meta_cameras[camera_id]["intrin"][0, 2], meta_cameras[camera_id]["intrin"][1, 2]

        #load image
        image_name = str(camera_id)+'.png'
        image_path = os.path.join(path_to_tsdir, str(camera_id)+'.png')
        _image = np.asarray(Image.open(image_path).resize(img_dim, resample=Image.BOX))

        height, width, _ = _image.shape
        print(height, width)
        camera = pyrender.IntrinsicsCamera(fx=focal_x*uf, fy=focal_y*uf, cx = (cx)*uf, cy = (cy)*uf, zfar=1e8*uf, znear = 0.01)
        # render the tracked mesh onto the camera
        mesh = pyrender.Mesh.from_trimesh(tracked_mesh)
        scene = pyrender.Scene(ambient_light = np.zeros(3), bg_color=[0.0, 0.0, 0.0])
        scene.add(mesh)
        scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(viewport_width = img_dim[0], viewport_height = img_dim[1])
        rendered_img, depth = r.render(scene, flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(rendered_img)
        plt.title("Rendered image")
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(_image)
        plt.title("GT")
        plt.suptitle(f"Camera ID: {camera_id}, {expName} at {timestamp}")
        plt.savefig("test.png", bbox_inches='tight', pad_inches=0.0)
        r.delete()
    
    
    
# uf = 2 # upscale factor (rendering_res = image_res * upscale factor)
# img_dim = (256*uf, 256*uf)

# # Load data
# bfm_trimesh = trimesh.load('../output/after_ColorparamEst.ply')
# bfm_errormap_trimesh = trimesh.load('../output/after_paramEst2_errorMap.ply')
# image_path = '../data/EURECOM_Kinect_Face_Dataset/' + str(person_id).zfill(4) + '/s1/RGB/rgb_' + str(person_id).zfill(4) + '_s1_' + expression + '.bmp'
# img = np.asarray(Image.open(image_path).resize(img_dim, resample=Image.BOX))

# # Define camera model
# camera_pose = np.array([[1,  0,  0,  0],
#                         [0, -1,  0,  0],
#                         [0,  0, -1,  0],
#                         [0,  0,  0,  1],])
# camera = pyrender.IntrinsicsCamera(fx = 525*uf, fy = 525*uf, cx = 127.5*uf, cy = 165.5*uf, zfar=2000*uf)

# # Render bfm mesh (We are no shading the face model here, just take color ov vertex)
# mesh = pyrender.Mesh.from_trimesh(bfm_trimesh)
# scene = pyrender.Scene(ambient_light=np.zeros(3), bg_color=[1.0, 1.0, 1,0])
# scene.add(mesh)
# scene.add(camera, pose=camera_pose)
# r = pyrender.OffscreenRenderer(img_dim[0], img_dim[1])
# color, _ = r.render(scene, flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT)
# # make white background pixel transparent
# bfm_img = color.copy()
# for i in range(img_dim[0]):
#     for j in range(img_dim[1]):
#         if np.array_equal(bfm_img[i,j], np.array([255, 255, 255, 255])):
#             bfm_img[i,j,3] = 0

# # generate image for geometric error map
# mesh = pyrender.Mesh.from_trimesh(bfm_errormap_trimesh)
# scene = pyrender.Scene()
# scene.add(mesh)
# scene.add(camera, pose=camera_pose)
# r = pyrender.OffscreenRenderer(img_dim[0], img_dim[1])
# geometric_error_map_img , _ = r.render(scene, flags = pyrender.RenderFlags.FLAT)

# # Generate Plot
# plt.figure(figsize=(24, 8), dpi=32)
# plt.rcParams.update({'font.size': 30})

# # Subplot 1: RGB image
# plt.subplot(1,3,1)
# plt.axis('off')
# plt.imshow(img)

# # Subplot 2: Geometric error map
# plt.subplot(1,3,2)
# plt.axis('off')
# # Define the custom colormap from red to blue
# custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_to_red', ['blue', 'red'], N=100)
# plt.imshow([[0, 4], [0, 4]], cmap = custom_cmap)
# plt.colorbar(label = 'Geometric Error [mm]', ticks=[0,4], orientation = 'horizontal', pad= 0, fraction = 0.19, shrink=0.7)
# plt.imshow(geometric_error_map_img[30:-50, 40:-40])

# # Subplot 3: Reconstructed face model as overly on RGB image
# plt.subplot(1,3,3)
# plt.axis('off')
# plt.imshow(img)
# plt.imshow(bfm_img, interpolation='none')
# plt.tight_layout(pad=0.0)

# if save_figure:
#     plt.savefig((str(person_id).zfill(4) + '_s1_' + expression + '_vis.png'), bbox_inches='tight', pad_inches=0.0)
# else:
#     plt.show()

# # face reenactment result
# bfm_trimesh = trimesh.load('../output/after_paramEst_FACEREANACMENT.ply')
# # Render bfm mesh (We are no shading the face model here, just take color ov vertex)
# mesh = pyrender.Mesh.from_trimesh(bfm_trimesh)
# scene = pyrender.Scene(ambient_light=np.zeros(3), bg_color=[1.0, 1.0, 1,0])
# scene.add(mesh)
# scene.add(camera, pose=camera_pose)
# r = pyrender.OffscreenRenderer(img_dim[0], img_dim[1])
# color, _ = r.render(scene, flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT)
# # make white background pixel transparent
# bfm_img = color.copy()
# for i in range(img_dim[0]):
#     for j in range(img_dim[1]):
#         if np.array_equal(bfm_img[i,j], np.array([255, 255, 255, 255])):
#             bfm_img[i,j,3] = 0

# plt.figure(figsize=(8, 8), dpi=64)
# plt.axis('off')
# plt.imshow(img)
# plt.imshow(bfm_img, interpolation='none')
# plt.tight_layout(pad=0.0)

# if save_figure:
#     plt.savefig("24_neutral_with_expression_from_44_smile.png", bbox_inches='tight', pad_inches=0.0)
# else:
#     plt.show()