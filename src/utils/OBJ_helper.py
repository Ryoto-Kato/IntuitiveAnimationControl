# Class for obj
import os
import sys
import numpy as np
import trimesh

class OBJ:
    def __init__(self, filename, swapyz=False):
        "Loads a Wavefront OBJ file."
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.file = filename

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                _v = [float(values[1]), float(values[2]), float(values[3])]
                # v = map(float, values[1:4])
                if swapyz:
                    _v = [float(values[1]), float(values[3]), float(values[2])]
                self.vertices.append(_v)
            elif values[0] == 'vn':
                
                _vn = [float(values[1]), float(values[2]), float(values[3])]
                # v = map(float, values[1:4])
                if swapyz:
                    _vn = [float(values[1]), float(values[3]), float(values[2])]
                self.normals.append(_vn)
            elif values[0] == 'vt':
                self.texcoords.append([float(values[1]), float(values[2])])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
    
    
    def getListVertsTris(self):
        tris = []
        for triangle in self.faces:
            # 0: triangle index list
            # 1: normals
            # 2: texture coordinate
            # 3: material configuration
            tris.append(triangle[0])
        verts = self.vertices
        return verts, tris
            
    def getNdarrayVertsTris(self, is_start_from1 = False):
        """
        returns
            verts: (N, 3) np.ndarray (float)
            tris: (m, 3) np.ndarray (int): indices into the verts array
        """
        verts = np.asarray(self.vertices)
        tris = []
        
        for triangle in self.faces:
            # 0: triangle index list
            # 1: normals
            # 2: texture coordinate
            # 3: material configuration
            tris.append(triangle[0])
        tris = np.asarray(tris)

        if is_start_from1:
            for triangle in tris:
                for i in range(3):
                    # print(triangle)
                    triangle[i] = triangle[i] - int(1)
                    
        return verts, tris

    @staticmethod
    def write_PointClouds(save_path, vertices, mesh_name):
        num_verts = int(len(vertices)/3)
        # print(num_verts)
        with open(os.path.join(save_path, mesh_name + '.obj'), 'w') as f:
            #write vertex coordinate
            for i in range(num_verts):
                line = 'v' +" "+str(vertices[i*3])+" "+str(vertices[i*3+1])+" "+str(vertices[i*3+2])
                f.write(line)
                f.write('\n')
        f.close()
        # print("check /dataset/multiface/tracked_mesh/result_point_clouds.obj")

    @staticmethod
    def write_OBJfile(reference_obj_file, save_path, vertices, name_Exp):
        num_verts = int(len(vertices)/3)
        # print(num_verts)
        copy_lines = []
        for line in open(reference_obj_file, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                continue
            else:
                copy_lines.append(line)
        fname = os.path.join(save_path, 'result_mesh'+name_Exp+'.obj')
        with open(fname, 'w') as f:
            #write vertex coordinate
            for i in range(num_verts):
                line = 'v' +" "+str(vertices[i*3])+" "+str(vertices[i*3+1])+" "+str(vertices[i*3+2])
                f.write(line)
                f.write('\n')
            #write vt. vn, f
            for cl in copy_lines:
                f.write(cl)
                # f.write('\n')
        f.close()
        # print("check /dataset/multiface/tracked_mesh/result_mesh"+name_Exp+".obj")
        return fname
    
    #@staticmethod
    #convert to PLY given OBJ and texture