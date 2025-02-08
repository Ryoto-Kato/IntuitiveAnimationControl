import os, sys
import pyvista as pv
import trimesh
from argparse import ArgumentParser, Namespace


from trame.app import get_server               # Entry point to trame
from trame.ui.vuetify import SinglePageLayout  # UI layout
from trame.widgets import vuetify, vtk  
from pyvista.trame.ui import plotter_ui

server = get_server()                          # Create/retrieve default server
server.client_type = "vue2"                    # Choose between vue2 and vue3
state, ctrl = server.state, server.controller  # Extract server state and controller


path_to_folder = os.path.join(os.getcwd(), os.pardir, "samples", "3dgs")
facemask_fname = "FaceMask_sample_subd2_face_trimesh.obj"
assert os.path.exists(path_to_folder)
path_to_facemask = os.path.join(path_to_folder, facemask_fname)
assert os.path.exists(path_to_facemask)

# if you use remote server, you need to use buffer for the offscreen rendering
# pv.start_xvfb()

p = pv.Plotter()

facemask = pv.read(path_to_facemask)
_ = p.add_mesh(facemask, color = [0.0, 0.5, 0.5], point_size = 2.0)

# second mesh
ply_file = "sample_subd2_face.ply"
path_to_plyfile = os.path.join(path_to_folder, ply_file)
assert os.path.exists(path_to_plyfile)
mesh = pv.read(path_to_plyfile)
# if you want to render only vertices, otherwise polygon mesh
# _mesh = pv.PolyData(mesh.points)

_ = p.add_mesh(mesh, color = [0.75, 0.75, 0.75])

p.camera_position = [(46.587771748532454, -110.97787747048358, 187.63653999519397),
                     (-1.23187255859375, 29.05982208251953, 1069.2713623046875),
                     (0.003916643970848481, -0.987577042560248, 0.15708674325976588)]

with SinglePageLayout(server) as layout:
    with layout.content:
        # Use PyVista's Trame UI helper method
        #  this will add UI controls
        view = plotter_ui(p)

if __name__ =="__main__":
    server.start()
