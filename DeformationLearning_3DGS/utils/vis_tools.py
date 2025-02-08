import pyvista as pv
import numpy as np
import cv2
import trimesh
from dreifus.pyvista import render_from_camera

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from pyvista.trame.ui import plotter_ui
import trimesh.visual

def VisMesh_FreeCamera(args):
    """
    input: args
        - path2mesh
        - off_screen
    """
    path_to_mesh = args.path2mesh
    off_screen = args.off_screen
    vis_point = args.vis_point

    pv.OFF_SCREEN = args.off_screen
    pv.start_xvfb()

    server = get_server()
    state, ctrl = server.state, server.controller

    p = pv.Plotter(off_screen=off_screen)
    tri_mesh = trimesh.load(path_to_mesh)
    print(tri_mesh)
    if type(tri_mesh) == trimesh.points.PointCloud or vis_point:
        print(type(tri_mesh))
        p.add_points(tri_mesh.vertices, color = "pink")
    else:
        mesh = pv.wrap(tri_mesh)
        # p.set_background("black")
        p.add_mesh(mesh, scalars=np.asarray(tri_mesh.visual.vertex_colors), style="wireframe", culling = "back")
        # p.add_mesh(mesh, color="blue", style="wireframe",culling = "none")

    with SinglePageLayout(server) as layout:
        with layout.content:
            # Use PyVista's Trame UI helper method
            #  this will add UI controls
            view = plotter_ui(p)

    return server

def VisTrimesh(tri_meshes, attributes=None, off_screen = True, camera_extrinsic = None):
    """
    input: args
        - path2mesh
        - off_screen
    """

    pv.OFF_SCREEN = off_screen
    server = get_server()
    server.client_type='vue2'
    state, ctrl = server.state, server.controller

    p = pv.Plotter(off_screen=off_screen)

    if camera_extrinsic!=None:
        p.camera_position=camera_extrinsic

    for tri_mesh, attribute in zip(tri_meshes, attributes):
        if type(tri_mesh) == trimesh.points.PointCloud:
            print(type(tri_mesh))
            if attribute == None:
                p.add_points(tri_mesh.vertices, color='red', point_size=1000)
            else:
                _ = p.add_points(
                    tri_mesh.vertices,
                    scalars=attribute,
                    render_points_as_spheres=True,
                    point_size=1,
                    cmap = 'jet',
                )
        else:
            mesh = pv.wrap(tri_mesh)
            if attribute.all()!=None:
                p.add_mesh(mesh, scalars=attribute, cmap = 'jet', opacity = 1.0)
            else:
                p.add_mesh(mesh, color = trimesh.visual.random_color(), opacity = 1.0)

    with SinglePageLayout(server) as layout:
        with layout.content:
            # Use PyVista's Trame UI helper method
            #  this will add UI controls
            view = plotter_ui(p)
            print(p.camera_position)

    return server

def VisPointsAttributes(points, attributes, cmap = 'jet', screenshot = False, title = "", flag_render_from_camera = False, window_size = [], intrinsic = None, extrinsic = None):
    pv.OFF_SCREEN = True

    server = get_server()
    state, ctrl = server.state, server.controller

    if flag_render_from_camera:
        p = pv.Plotter(window_size=window_size, off_screen=True)
    else:
        p = pv.Plotter(off_screen = True)
        cmap = cmap + "_r"
        p.camera_position = [(46.587771748532454, -110.97787747048358, 187.63653999519397),
                             (-1.23187255859375, 29.05982208251953, 1069.2713623046875),
                             (0.003916643970848481, -0.987577042560248, 0.15708674325976588)]

    p.background_color = "white"
    _ = p.add_points(
        points,
        scalars=attributes,
        render_points_as_spheres=True,
        point_size=20,
        cmap = cmap,
    )
    if not flag_render_from_camera:
        if screenshot:
            p.save_graphic(title+"_screenshot.pdf")
        else:
            with SinglePageLayout(server) as layout:
                with layout.content:
                    # Use PyVista's Trame UI helper method
                    #  this will add UI controls
                    view = plotter_ui(p)

            return server
    else:
        rendered_image = render_from_camera(p, extrinsic, intrinsic)
        cv2.imwrite(title+"_screenshot.png", rendered_image)

def VisPointsWithColor(points, perVertexColor, cmap = 'jet', screenshot=False, title = ""):
    p = pv.Plotter(off_screen=True)
    p.camera_position = [(46.587771748532454, -110.97787747048358, 187.63653999519397),
    (-1.23187255859375, 29.05982208251953, 1069.2713623046875),
    (0.003916643970848481, -0.987577042560248, 0.15708674325976588)]

    color = ((perVertexColor/np.max(perVertexColor, axis = 0))*255)
    
    _ = p.add_points(
        points,
        vertex_color = color,
        render_points_as_spheres=True,
        point_size=5,
    )
    if screenshot:
        p.save_graphic(title+"screenshot.pdf")
    else:
        p.show()