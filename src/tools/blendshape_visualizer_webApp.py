import sys
import os
import numpy as np

# get current date and year
from datetime import datetime
now = datetime.now()
date = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")
# name the folder to write/read .obj while visualization
exp_name = "exp"+date

# for web app
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# append src directory to the sys.path
path_to_src = os.pardir
print(path_to_src)
sys.path.append(path_to_src)

# for blendshape
from utils.pickel_io import load_from_memory

from utils.Blendshape import Blendshape, datastruct_blendshape
from utils.Routine_pyvista import Routine_blendshape

from utils.OBJ_helper import OBJ

# debug mode
DEBUG = False

def input_error_print():
    print("-"*10)
    print("[ERROR] run with 'python *.py [pkl_file_name] --server'")
    print("\tWithout specification, based on following .pkl")
    print("-"*10)


# read args
try:
    pickel_fname = sys.argv[1]
    if (pickel_fname == "--server"):
        input_error_print()
except:
    input_error_print()


# configure the fname of .pkl
# pickel_fname = "blendshape_26102023_17_21.pkl"

path_to_memory = os.path.join(os.getcwd(), os.pardir, os.pardir, "dataset", "memory", "test-multiface", "tracked_mesh", "memory")
# print(path_to_memory)

# path to folder which stores .obj
save_path = os.path.join(os.getcwd(), os.pardir, os.pardir,'dataset', 'multiface', 'tracked_mesh')
# sample path to write obj file with copying vt, f lines since they are topologically equivalent
sample_path = os.path.join(save_path, "sample.obj")

# load blendshape data 
datastruct_blendshape = load_from_memory(path_to_memory=save_path, pickle_fname=pickel_fname)

# create a folder [name: exp(date)] to write and read .obj file to be visualized
exp_folder_path = os.path.join(save_path, exp_name)
if not os.path.exists(exp_folder_path):
    os.mkdir(exp_folder_path)

# configure the blendshape components
_MEAN = datastruct_blendshape.MEAN
_PCs = datastruct_blendshape.PCs
_Stds = datastruct_blendshape.Stds
_save_path = exp_folder_path
_D = datastruct_blendshape.PCs.shape[0]
_only_specific_pc = False
_name_newExp = ""

# initialize blendshape class, and instance is created
blendshape = Blendshape(_MEAN, _PCs, _Stds, _D, _save_path, _only_specific_pc, _name_newExp)

# average-exp mesh file name
ave_fname = blendshape.generate_aveExp()

# Always set PyVista to plot off screen with Trame
pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller


# test with cube mesh
# mesh = pv.Wavelet()

# p = pv.Plotter()
# p.add_mesh(mesh)

# blendshape visualization setting

engine = Routine_blendshape(blendshape)
p = pv.Plotter()

p.camera_position = [(46.587771748532454, -110.97787747048358, 187.63653999519397),
 (-3.23187255859375, 29.05982208251953, 1069.2713623046875),
 (0.003916643970848481, -0.987577042560248, 0.15708674325976588)]

# create UIs
num_blendshapes_tobevisualized = 10

step_size = (0.95-0.05)/num_blendshapes_tobevisualized
step_y = np.flip(np.arange(0.05, 0.95, step_size, dtype = float))
p.enable_parallel_projection()
# sargs = dict(interactive=True) 
sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)
p.add_mesh(engine.starting_mesh, scalars="distances", show_edges=False, smooth_shading=True, scalar_bar_args=sargs)

p.add_slider_widget(
    callback=lambda value: engine(str(0), value, DEBUG),
    rng=[0, 1*blendshape.Stds[0]],
    value=0.0,
    title="blendshape"+str(0),
    pointa=(0.8, step_y[0]),
    pointb=(0.95, step_y[0]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(1), value, DEBUG),
    rng=[0, 1*blendshape.Stds[1]],
    value=0.0,
    title="blendshape"+str(1),
    pointa=(0.8, step_y[1]),
    pointb=(0.95, step_y[1]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(2), value, DEBUG),
    rng=[0, 1*blendshape.Stds[2]],
    value=0.0,
    title="blendshape"+str(2),
    pointa=(0.8, step_y[2]),
    pointb=(0.95, step_y[2]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(3), value, DEBUG),
    rng=[0, 1*blendshape.Stds[3]],
    value=0.0,
    title="blendshape"+str(3),
    pointa=(0.8, step_y[3]),
    pointb=(0.95, step_y[3]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(4), value, DEBUG),
    rng=[0, 1*blendshape.Stds[4]],
    value=0.0,
    title="blendshape"+str(4),
    pointa=(0.8, step_y[4]),
    pointb=(0.95, step_y[4]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(5), value, DEBUG),
    rng=[0, 1*blendshape.Stds[5]],
    value=0.0,
    title="blendshape"+str(5),
    pointa=(0.8, step_y[5]),
    pointb=(0.95, step_y[5]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(6), value, DEBUG),
    rng=[0, 1*blendshape.Stds[6]],
    value=0.0,
    title="blendshape"+str(6),
    pointa=(0.8, step_y[6]),
    pointb=(0.95, step_y[6]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(7), value, DEBUG),
    rng=[0, 1*blendshape.Stds[7]],
    value=0.0,
    title="blendshape"+str(7),
    pointa=(0.8, step_y[7]),
    pointb=(0.95, step_y[7]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(8), value, DEBUG),
    rng=[0, 1*blendshape.Stds[8]],
    value=0.0,
    title="blendshape"+str(8),
    pointa=(0.8, step_y[8]),
    pointb=(0.95, step_y[8]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

p.add_slider_widget(
    callback=lambda value: engine(str(9), value, DEBUG),
    rng=[0, 1*blendshape.Stds[9]],
    value=0.0,
    title="blendshape"+str(9),
    pointa=(0.8, step_y[9]),
    pointb=(0.95, step_y[9]),
    style='modern',
    slider_width = 0.02,
    tube_width = 0.01,
    title_height = 0.02,
)

with SinglePageLayout(server) as layout:
        
    with layout.content:
        # Use PyVista's Trame UI helper method
        #  this will add UI controls
        view = plotter_ui(p)

if __name__ == "__main__":
    server.start()

