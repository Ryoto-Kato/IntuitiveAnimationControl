import numpy as np
import json

def load_RT(path):
    """
    0.9705157 0.035186626 0.2384557 -270.26154
    -0.1084007 0.9473142 0.30140525 -291.6677
    -0.21528703 -0.3183673 0.92319757 121.37361    
    """

    extrin = []

    with open(path, "r") as f:
        extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
    
    return np.array(extrin)


def load_KRT(path):
    """
    Data loader for the following KRT data

    camera name

    [intrinsic matrix 3x3]
    [dist 3]
    [extrinsic matrix 3x4]

    like following

    400012
    
    7724.681 0.0 797.0469
    0.0 7724.834 924.32745
    0.0 0.0 1.0
    0.0 0.0 0.0 0.0 0.0
    0.9705157 0.035186626 0.2384557 -270.26154
    -0.1084007 0.9473142 0.30140525 -291.6677
    -0.21528703 -0.3183673 0.92319757 121.37361Pose
    """
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]

            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

        return cameras

def load_KRT_from_json(path2camjson):
    """
    input
        args.path2camjson: path to json which describe the camera extrinsic and intrinsic
    output
        dictionary
    """

    print(f"loading KRT json from path {path2camjson}")
    with open(path2camjson, 'r') as camjson:
        camera_configs = json.load(camjson)

    cameras = {}
    _intrinsic = camera_configs["intrinsics"]
    # print(f"intrinsic: {_intrinsic}")

    for cam_id in camera_configs["world_2_cam"].keys():
        intrin = _intrinsic
        extrin = camera_configs["world_2_cam"][cam_id]

        cameras[cam_id] = {
            "intrin": np.array(intrin),
            "dist": np.zeros(4, dtype=float),
            "extrin": np.array(extrin),
            "config": {"intrin": "OPENCV", "extrin": "W2C"}
        }
    
    sorted_cameras = dict(sorted(cameras.items()))
         
    return sorted_cameras