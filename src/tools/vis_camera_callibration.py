import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.matrix import Pose, Intrinsics, CameraCoordinateConvention, PoseType
import sys, os
import numpy as np
import vtk
import trimesh
from PIL import Image
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from pyvista.trame.ui import plotter_ui

# append src directory to the sys.path
sys.path.append(os.pardir)

from utils.metadata_loader import load_KRT, load_RT
from utils.Dataset_handler import Filehandler

def input_error_print():
    print("-"*10)
    print("[ERROR] run with 'python *.py KRT [Expression name e.g., E001_Neutral_Eyes_Open] --server'")
    print("\tWithout specification, based on following .pkl")
    print("-"*10)

# read args
try:
    f_KRT = sys.argv[1]
    expName = sys.argv[2]
    # time_stamp = sys.argv[3]

    if (f_KRT == "--server" or expName == "--server"):
        input_error_print()
except:
    input_error_print

# Always set PyVista to plot off screen with Trame
pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

f_KRT = "KRT"
path_to_meta = os.path.join(os.getcwd(), os.pardir, os.pardir, "dataset", "multiface", "meta_data")
cameras = load_KRT(os.path.join(path_to_meta,f_KRT))

# path to data to be visualized
mesh_directory = os.path.join(os.getcwd(), os.pardir, os.pardir,"dataset", "multiface", "tracked_mesh", expName)
image_directory = os.path.join(os.getcwd(), os.pardir, os.pardir, "dataset", "multiface", "images", expName)

# Load dataset and find first frame name
# timestamp
list_dirNames, list_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=image_directory, prefix='4000')
list_fNames, list_fPaths = Filehandler.fileWalker_InDirectory(path_to_directory=list_dirPaths[0], ext='.png')
# print(len(list_of_directories)
time_stamp = list_fNames[0][:6]
print(f"Show the images at {time_stamp}")

tri_mesh = trimesh.load(os.path.join(mesh_directory, str(time_stamp)+".obj"))
head_pose = load_RT(os.path.join(mesh_directory, str(time_stamp)+"_transform.txt")) #Canonical 2 World

print("------------------------------")
print("Head pose")
print(head_pose.shape)
print(head_pose)
print("------------------------------")


p = pv.Plotter()
mesh = pv.wrap(tri_mesh)
p.add_mesh(mesh)
counter = 0
# cameras which will be used for 3D Gaussians
# cameras_id = [400004, 400026, 400063, 400060, 400013, 400016, 400049, 400069]
print("Loading camera callibration...")
print("Number of Cameras: ", len(cameras))

# CAM2WORLD
# rightCams = [400064, 400023, 400029, 400013, 400048, 400037, 400007, 400027, 400026, 400017, 400004, 400028, 400018, 400059, 400067, 400010, 400008, 400055]
# WORLD2CAM
# leftCams = [400013, 400029, 400048, 400037, 400007, 400008, 400055, 400023, 400064, 400049, 400031, 400027, 400026, 400017, 400004, 400028, 400018, 400059, 400050, 400010, 400055]
# print("Left side cameras information")
# print("Number of camera at the left side: ", len(leftCams))

for key in cameras.keys():
    camera_extrinsic = np.vstack((cameras[key]["extrin"], [0,0,0,1]))
    camera_intrinsic = cameras[key]["intrin"]
    pose = Pose(camera_extrinsic, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
    image = Image.open(os.path.join(image_directory, key, time_stamp+".png"))
    intrinsics = Intrinsics(camera_intrinsic)
    add_camera_frustum(p=p, pose=pose, intrinsics=intrinsics, image=np.array(image), label=key, size=3e2)
    print("--------------------------------------")
    print(f"key: {key}")
    print(f"camera_extrinsic\n {camera_extrinsic}")
    print(f"camera_intrinsic\n {camera_intrinsic}")
    print(f"image shape\n {np.asarray(image).shape}")
    counter = counter +1


with SinglePageLayout(server) as layout:
    with layout.content:
        # Use PyVista's Trame UI helper method
        #  this will add UI controls
        view = plotter_ui(p)

if __name__ == "__main__":
    server.start()
    # p.show()
    


