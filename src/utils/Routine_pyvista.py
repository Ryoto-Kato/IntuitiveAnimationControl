from .Blendshape import Blendshape
import pyvista as pv
import numpy as np

class Routine_blendshape:
    def __init__(self, blendshape:Blendshape):
        self.blendshape = blendshape
        # Expected pyvista mesh type
        self.ave_fname = self.blendshape.generate_aveExp()
        self.ori_ave_mesh = pv.read(self.ave_fname)
        self.ave_mesh = pv.read(self.ave_fname)
        self.starting_mesh = self.ori_ave_mesh
        self.starting_mesh["distances"] = self.L2_norm(self.ori_ave_mesh)
        self.output = self.starting_mesh

    def __call__(self, index, value, debug:bool):
        self.blendshape.set_individual_coefficient(int(index), value, debug=debug)
        self.update()

    def L2_norm(self, new_exp):
        # calculate L2 distance between average_exp and new_exp
        closest_cells, closest_points = self.ave_mesh.find_closest_cell(new_exp.points, return_closest_point=True)
        #take L2 norm between a closest spatial point of ave_exp and new_exp
        d_exact = np.linalg.norm(new_exp.points - closest_points, axis=1)
        return d_exact

    def update(self):
        self.blendshape.get_newExp()
        result_f = self.blendshape.generate_newExp()
        result_mesh = pv.read(result_f)
        result_mesh["distances"] = self.L2_norm(result_mesh)
        self.output.copy_from(result_mesh)
        return