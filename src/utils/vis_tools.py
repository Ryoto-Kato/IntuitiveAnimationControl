import pyvista as pv
import numpy as np
import cv2
from dreifus.pyvista import render_from_camera

def VisPointsAttributes(points, attributes, cmap = 'jet', screenshot = False, title = "", flag_render_from_camera = False, window_size = [], intrinsic = None, extrinsic = None):
    pv.start_xvfb()
    if flag_render_from_camera:
        p = pv.Plotter(window_size=window_size, off_screen=True)
    else:
        p = pv.Plotter()
        cmap = cmap + "_r"
        p.camera_position = [(46.587771748532454, -110.97787747048358, 187.63653999519397),
                             (-1.23187255859375, 29.05982208251953, 1069.2713623046875),
                             (0.003916643970848481, -0.987577042560248, 0.15708674325976588)]

    p.background_color = "black"
    _ = p.add_points(
        points,
        scalars=attributes,
        render_points_as_spheres=True,
        point_size=20,
        cmap = cmap,
    )
    if not flag_render_from_camera:
        if screenshot:
            p.save_graphic(title+"_screenshot.png")
        else:
            p.show()
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