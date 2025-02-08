import os,sys
import numpy as np
from typing import NamedTuple

# utils of 3DGS
from utils.graphics_utils import focal2fov
from scene.dataset_readers import CameraInfo, getNerfppNorm, SceneInfo, fetchPly

# utils of 3DSSL-WS23_IntuitiveAnimation
path_to_3WI = os.path.join(os.getcwd(), os.pardir, "3DSSL-WS23_IntuitiveAnimation", "")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.Dataset_handler import Filehandler
from utils.metadata_loader import load_KRT, load_RT
from utils.PLY_helper import tex2vertsColor, gammaCorrect
from PIL import Image

color_corr = lambda x: (255 * gammaCorrect(x / 255.0, dim=2)).clip(0, 255)

def readMultifaceSceneInfo(expName, frame_counter = 0, ALLcam=False):
    ID = '6795937'
    Neutral_exp = "E001_Neutral_Eyes_Open"
    first_time_stamp = "000102"
    
    # [TODO] set the path to re-structured mutl-face dataset
    # path_to_dataset = os.path.join(os.getcwd(), os.pardir, "dataset")
    path_to_dataset = os.path.join(os.pardir, "dataset")
    path_to_multiface = os.path.join(path_to_dataset, "multiface")
    path_to_metadata = os.path.join(path_to_multiface, "meta_data")

    # Get the directory containing images at each time stamp in the expression folder 
    list_TimeStampDirNames, list_TimeStampDir_Paths = Filehandler.dirwalker_InFolder(path_to_folder=os.path.join(path_to_dataset, "multi_views", str(ID), expName), prefix='0')

    # Get the name of first frame time stamp
    time_stamp = list_TimeStampDirNames[frame_counter][:6]
    path_to_tsdir = list_TimeStampDir_Paths[frame_counter]
    print("Time stamp: ", time_stamp)

    # path to original tracked mesh 
    # ply_path = os.path.join(path_to_tsdir, time_stamp+'.ply')
    
    # path to subdivided ply instead of originall tracked mesh
    # ply_path = os.path.join(path_to_tsdir, time_stamp+'_subd.ply')
    ply_path = os.path.join(path_to_tsdir, time_stamp+'_subd2.ply')

    transform_path = os.path.join(path_to_tsdir, time_stamp+"_transform.txt")

    head_pose = load_RT(transform_path)

    ID = 6795937
    f_KRT = "KRT"
    meta_cameras = load_KRT(os.path.join(path_to_metadata, f_KRT))
    # eval
    # eval = False
    llffhold = 8 # where i%8==0: test_cam_infos = cam_infos[i]  

    Train_SelectedCAM = meta_cameras.keys()

    cam_infos = []
    
    for i, cam_name in enumerate(meta_cameras.keys()):
        camera_id = cam_name
        
        sys.stdout.write("loading camera {}/{}".format(i+1, len(meta_cameras.keys())))
        sys.stdout.write("_Name: {}".format(cam_name))
        sys.stdout.flush()

        # load extrinsic
        trans_R = np.transpose(meta_cameras[camera_id]["extrin"][:3, :3]) #W2C to C2W
        T = np.array(meta_cameras[camera_id]["extrin"][:, 3:4]).ravel()
        # load intrinsic
        focal_x, focal_y = meta_cameras[camera_id]["intrin"][0, 0], meta_cameras[camera_id]["intrin"][1, 1]
        cx, cy = meta_cameras[camera_id]["intrin"][0, 2], meta_cameras[camera_id]["intrin"][1, 2]
        # load image
        image_name = str(cam_name)+'.png'
        image_path = os.path.join(path_to_tsdir, str(cam_name)+'.png')
        _image = Image.open(image_path)

        # backgroud remover
        np_image = np.array(_image)
        correct_image = color_corr(np_image)
        _PIL_correct_image = Image.fromarray(np.uint8(correct_image))
        # print(_PIL_correct_image.size)
        
        # width, height
        width, height = _PIL_correct_image.size
        print(f"- image size (w, h): {width, height}")
        
        # FOV
        FovY = focal2fov(focal_y, height)
        FovX = focal2fov(focal_x, width)

        # Camera Infos
        cam_info = CameraInfo(uid=camera_id, R = trans_R, T = T, FovY = FovY, FovX = FovX, image = _PIL_correct_image, image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
        sys.stdout.write('\n')

    # Sort camera infos according to the name of picture
    cam_infos = sorted(cam_infos.copy(), key=lambda x : x.image_name)

    if not ALLcam:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos

    print("Num of train_cams:", len(train_cam_infos))
    # print("Train cams: ", train_cam_infos)
    print("Num of test_cams:", len(test_cam_infos))
    # print("Test cams: ", test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = fetchPly(ply_path)

    print(pcd.points)
    print(pcd.colors.shape)
    print(pcd.normals.shape)

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)

    return scene_info

