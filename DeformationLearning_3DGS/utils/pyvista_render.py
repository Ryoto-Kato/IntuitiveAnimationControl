import numpy as np
import sys, os
import trimesh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

#dreifus
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum, render_from_camera
from dreifus.matrix import Pose, Intrinsics, CameraCoordinateConvention, PoseType

# pyvista and trame
import pyvista as pv
from trame.app import get_server               # Entry point to trame
from trame.ui.vuetify import SinglePageLayout  # UI layout
from trame.widgets import vuetify, vtk  
from pyvista.trame.ui import plotter_ui

from argparse import ArgumentParser, Namespace

server = get_server()                          # Create/retrieve default server
server.client_type = "vue2"                    # Choose between vue2 and vue3
state, ctrl = server.state, server.controller  # Extract server state and controller

path_to_3WI = os.path.join(os.getcwd(), os.pardir, os.pardir)
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.Dataset_handler import Filehandler
from utils.metadata_loader import load_KRT, load_RT
from utils.PLY_helper import tex2vertsColor, gammaCorrect

color_corr = lambda x: (255 * gammaCorrect(x / 255.0, dim=2)).clip(0, 255)
clamp01 = lambda x : (x/(np.max(x)-np.min(x)))

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
# python pyvista_render.py --expName=E001_Neutral_Eyes_Open  --savefig

path_to_dataset = os.path.join(path_to_3WI, "dataset")
path_to_multiface = os.path.join(path_to_dataset, "multiface")
path_to_metadata = os.path.join(path_to_multiface, "meta_data")
path_to_output = os.path.join(os.getcwd(), os.pardir, "output", "projection_test")

if not os.path.exists(path_to_output):
    os.mkdir(path_to_output)

savefolder_name = str(ID)+ "_" +str(expName)+ "_" +str(timestamp)

path_to_savefolder = os.path.join(path_to_output, savefolder_name)
if not os.path.exists(path_to_savefolder):
    os.mkdir(path_to_savefolder)

# Get the directory containing images at each time stamp in the expression folder
list_TimeStampDirNames, list_TimeStampDir_Paths = Filehandler.dirwalker_InFolder(path_to_folder = os.path.join(path_to_dataset, "multi_views", ID, expName), prefix='0')

if timestamp == None:
    timestamp = list_TimeStampDirNames[0][:6]
print(f"Expression:{expName}, Timestamp:{timestamp}")

path_to_tsdir = list_TimeStampDir_Paths[0]

assert os.path.exists(path_to_tsdir) == True

obj_path = os.path.join(path_to_tsdir, timestamp+'.obj')
tracked_mesh = trimesh.load(file_obj = obj_path)
print(tracked_mesh)

# for i, point in enumerate(tracked_mesh.vertices):
#     # print("before:",point)
#     tracked_mesh.vertices[i][1] *= -1
#     tracked_mesh.vertices[i][2] *= -1
#     # print("after:" ,tracked_mesh.vertices[i])

# virtual frame buffer for offscreen rendering
pv.start_xvfb()

img_dim = (1334, 2048) # width and height

# pvista scene setting
p = pv.Plotter(window_size = [img_dim[0], img_dim[1]], off_screen = True)

# background
p.background_color = (0, 0, 0, 1.0)

# add mesh in the scene
mesh  = pv.wrap(tracked_mesh)
p.add_mesh(mesh)

headpose = load_RT(os.path.join(path_to_tsdir, timestamp+"_transform.txt"))
f_KRT = "KRT"
meta_cameras = load_KRT(os.path.join(path_to_metadata, f_KRT))

for i, cam_name in enumerate(meta_cameras.keys()):
    camera_id = cam_name

    sys.stdout.write("Reading camera {}/{}\n".format(i+1, len(meta_cameras.keys())))
    sys.stdout.write("Name: {}\n".format(cam_name))
    sys.stdout.flush()

    #load extrinsic
    camera_extrin = np.vstack((meta_cameras[camera_id]["extrin"], [0,0,0,1]))
    camera_pose = Pose(camera_extrin, pose_type = PoseType.WORLD_2_CAM)
    
    # load intrinsic
    intrinsics = Intrinsics(meta_cameras[camera_id]["intrin"])
    focal_x, focal_y = meta_cameras[camera_id]["intrin"][0, 0], meta_cameras[camera_id]["intrin"][1, 1]
    cx, cy = meta_cameras[camera_id]["intrin"][0, 2], meta_cameras[camera_id]["intrin"][1, 2]

    #load image
    image_name = str(camera_id)+'.png'
    image_path = os.path.join(path_to_tsdir, str(camera_id)+'.png')
    gt_image = np.asarray(Image.open(image_path).resize(img_dim, resample=Image.Resampling.BOX))
    bg_image = np.zeros_like(gt_image)
    correct_gt_image = color_corr(gt_image)
    _PIL_correct_image = clamp01(Image.fromarray(np.uint8(correct_gt_image)))

    height, width, _ = gt_image.shape
    print(height, width)
    # rendered_img = None
    rendered_img = render_from_camera(p, camera_pose, intrinsics)
    rendered_img = clamp01(rendered_img)
    
    rendered_img = clamp01(bg_image + rendered_img[..., :3])

    # plt.subplot(1, 2, 1)
    # plt.axis('off')
    # plt.imshow(rendered_img)
    # plt.title("Rendered image")
    # plt.subplot(1, 2, 2)
    # plt.axis('off')
    # plt.imshow(_PIL_correct_image)
    # plt.title("GT")
    # plt.suptitle(f"Camera ID: {camera_id}, {expName} at {timestamp}")
    plt.imshow(_PIL_correct_image)
    plt.axis('off')
    fname = os.path.join(path_to_savefolder, str(camera_id)+".png")
    sys.stdout.write("Save rendered image: {}\n".format(fname))
    plt.savefig(fname, dpi = 64)

