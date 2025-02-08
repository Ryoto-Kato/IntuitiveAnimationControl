from dataclasses import dataclass
import numpy as np
import trimesh

@dataclass
class PCA_MBSPCA_SLDC:
    dataMat: np.ndarray # centralized and normalized datamatrix
    pca_C: np.ndarray # not scaled yet
    pca_W:np.ndarray # not scaled yet
    mbspca_C: np.ndarray # not scaled yet
    mbspca_W: np.ndarray # not scaled yet
    sldc_C: np.ndarray
    sldc_W:np.ndarray


def attribute_subd1(attribute, sample_mesh):
    """
    sample_mesh: should be mesh in trimesh data structure (5509 verts)
    attribute: attribute of each vertices (attribute should be in shape [5509, #attribs])
    """
    subd1_vertices, subd1_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((sample_mesh.vertices, attribute)),
        faces = sample_mesh.faces
        )

    final_phi = subd1_vertices[:, 3:]
    final_verts = subd1_vertices[:, :3]
    return final_phi, final_verts

def attribute_subd2(attribute, sample_mesh):
    """
    sample_mesh: should be mesh in trimesh data structure (5509 verts)
    attribute: attribute of each vertices (attribute should be in shape [5509, #attribs])
    """
    subd1_vertices, subd1_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((sample_mesh.vertices, attribute)),
        faces = sample_mesh.faces
        )

    subd1_verts = subd1_vertices[:, :3]
    subd1_phiheat = subd1_vertices[:, 3:]

    subd2_vertices, subd2_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((subd1_verts, subd1_phiheat)),
        faces = subd1_faces
    )

    final_phi = subd2_vertices[:, 3:]
    final_verts = subd2_vertices[:, :3]
    return final_phi, final_verts

def attribute_subd2(attribute, sample_mesh):
    """
    sample_mesh: should be mesh in trimesh data structure (5509 verts)
    attribute: attribute of each vertices (attribute should be in shape [5509, #attribs])
    """
    subd1_vertices, subd1_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((sample_mesh.vertices, attribute)),
        faces = sample_mesh.faces
        )

    subd1_verts = subd1_vertices[:, :3]
    subd1_phiheat = subd1_vertices[:, 3:]

    subd2_vertices, subd2_faces = trimesh.remesh.subdivide(
        vertices=np.hstack((subd1_verts, subd1_phiheat)),
        faces = subd1_faces
    )

    final_phi = subd2_vertices[:, 3:]
    final_verts = subd2_vertices[:, :3]
    return final_phi, final_verts