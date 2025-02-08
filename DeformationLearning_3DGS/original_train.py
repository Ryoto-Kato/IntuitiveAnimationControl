import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from train import training

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# [TODO] set the path to "3DSSL-WS23_IntuitiveAnimation"
path_to_3WI = os.path.join(os.getcwd(), os.pardir)
sys.path.append(os.path.join(path_to_3WI, 'src'))

from utils.OBJ_helper import OBJ
from utils.Dataset_handler import Filehandler
from utils.pickel_io import dump_pckl

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # load hyper parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    max_iterations =  10_000
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(np.arange(0, max_iterations, 2500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(np.arange(0, max_iterations, 10000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expID", type=int, default=0)
    parser.add_argument("--ALLcam", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.iterations = 10_000
    args.random_background = True
    args.save_iterations.append(args.iterations)

    #[TODO] set the appropriate path to dataset/output
    ID = '6795937'
    # path_to_dataset = os.path.join(path_to_3WI, "dataset")
    path_to_dataset = "../dataset"
    path_to_output = "/mnt/hdd/output/multiface"
    path_to_MultiViews = os.path.join(path_to_dataset, 'multi_views')
    path_to_ID = os.path.join(path_to_MultiViews, ID)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    list_expNames, list_expPaths =  Filehandler.dirwalker_InFolder(path_to_folder=path_to_ID, prefix='E0')
    
    _ALLcam = args.ALLcam

    unique_id = str(uuid.uuid4())
    if _ALLcam:
        unique_str = unique_id[:10] + "ALLcam"
    else:
        unique_str = unique_id[:10] + "notALLcam" #33 cameras for training and 5 cameras for evaluation

    numOfFrames = 1

    for i, _expName in enumerate(list_expNames):
        print("Optimization: " + _expName)
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, subject_id = ID, expName=_expName, unique_str=unique_str, ALLcam=_ALLcam, NofFrame=numOfFrames, path_to_output=path_to_output)
        
    # All done
    print("\nTraining complete.")



