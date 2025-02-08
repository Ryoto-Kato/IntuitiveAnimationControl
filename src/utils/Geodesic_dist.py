import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from utils.common_utils import vecnorm, normalized
from utils.vis_tools import VisPointsAttributes

import trimesh
import networkx as nx

class GeodesicDistSimple(object):
    """
    compute geodisc distance by tracing shortest pathes (edges)
    """
    def __init__(self, verts, tris):
        self._verts = verts
        self._tris = tris
        self.mesh = trimesh.Trimesh(vertices=self._verts, faces=self._tris, process=False)
        self.edges = self.mesh.edges_unique
        self.length = self.mesh.edges_unique_length
        self.g = nx.Graph()
        self.pathes = []
        self.distances = []
        
    
    def __call__(self, idx):
        for edge, L in zip(self.edges, self.length):
            self.g.add_edge(*edge, length =L)
        
        pathes = []
        distances = []
        for i in range(len(self.mesh.vertices)):
            # path = nx.shortest_path(self.g, source=idx, target=i, weight = 'length')
            dist = nx.shortest_path_length(self.g, source=idx, target=i, weight= 'length')
            # pathes.append(path)
            distances.append(dist)
        self.pathes = pathes
        self.distances = distances
        return distances
        
    def visualize_distance_func(self, cmap = 'jet'):
        VisPointsAttributes(self._verts, self.distances, cmap)


def compute_support_map(phi, min_dist, max_dist):
    """
    Arguments
        idx: index of source vertex
        geodesic_func: instance of GeodesicDistHeatMethod()
        min_dist: minimum distance, dist<=min_dist = 0
        max_dist: maximum distance, dist>=max_dist = 1
    Returns
        cofficient assignment for vertices: np.ndarray
        
        dist<=min_dist =>0
        dist>=max_dist =>1
        otherwise (dist - min_dist) / (max_dist - min_dist) 
    """
    coeffs = (np.clip(phi, min_dist, max_dist) - min_dist) / (max_dist - min_dist)
    coeffs[coeffs>=max_dist] = 1.0
    coeffs[coeffs<=min_dist] = 0.0
    return coeffs

def compute_support_map_gauss(phi, mu, sigma, scaler=1.0):
    """
    Arguments

    Returns
        a*exp(-1*c)
    """
    a = 1/(sigma * np.sqrt(2 * np.pi))
    c = 0.5 * ((phi-mu)/sigma)**2
    coeffs = a * np.exp(-1 * c)
    # coeffs = scaler * (coeffs/np.max(coeffs))
    coeffs = np.max(coeffs) - (coeffs)
    # print("max coeff: ", np.max(coeffs))
    return coeffs

def compute_topological_laplacian(verts, tris):
    """ran
    Compute topological laplacian given vertices and triangles

    Input:
        tris: triangle index list (index should be [0, ...])
        vertex: vertex list
    
    Output:
        L: contangent weight matrix 
            compute cotangent weight at each vertex and store in sparse matrix L
            - Diagonal entries are sum of value in the column
            - cotangent weight is computed by following.
                - w_ij = .5*cot(a_ij) + .5*cot(b_ij)
                - a_ij = the angle which is at opposite of edge i---j
        
        VA: area per vertex matrix
            compute a area per vertex by dividing area of triangle (.5*cross product) by 3 and store them as diagonal entries in matrix VA
    """

    if(tris.min() > 0):
        for triangle in tris:
            for i in range(3):
                triangle[i] = triangle[i] - int(1)

    N = len(verts)
    W_ij = np.empty(0, np.double)
    I = np.empty(0, np.int32)
    J = np.empty(0, np.int32)

    for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        # get vertex id list
        vi1 = tris[:, i1] 
        vi2 = tris[:, i2]
        vi3 = tris[:, i3]

        # compute vector from vi1 to vi2 (vector u) and to vi3 (vector v)
        u = verts[vi2] - verts[vi1]
        v = verts[vi3] - verts[vi1]

        # cotangent
        cotan = (u*v).sum(axis = 1) / vecnorm(np.cross(u, v))
        # Each W_ij has the cotangent at the vertex which is at opposite to edge i---j
        W_ij = np.append(W_ij, 0.5 * cotan)
        I = np.append(I, vi2)
        J = np.append(J, vi3)
        W_ij = np.append(W_ij, 0.5*cotan)
        I = np.append(I, vi3)
        J = np.append(J, vi2)
    # Create a sparse matrix
    L  = sparse.csr_matrix((W_ij, (I, J)), shape = (N, N))
    # L * np.ones(N) = L.sum(axis = 1)
    L = L - sparse.spdiags(L * np.ones(N), 0, N, N)

    # If the L has non-element row/column, L is singular
    # L should be invertible, we insert -1 where diagonal entry is 0

    L = L.toarray()
    for i in range(L.shape[1]):
        if L[i][i] == 0:
            L[i][i] = -1

    # convert matrix L into Compressed Sparse Row format
    L = sparse.csr_matrix(L)

    # triangle area matrix
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]

    # compute normal at each triangle
    n = np.cross(e1, e2)

    # triangle are can be computed by
    triangle_area = 0.5 * vecnorm(n)

    # compute per-vertex area
    vertex_area = np.zeros(N)

    # triangle area
    ta3 = triangle_area / 3

    # construct vertex area matrix VA
    for i in range(tris.shape[1]):
        # count the number of occurence of index in the list of triangle
        bc = np.bincount(tris[:, i].astype(int), ta3)
        vertex_area += bc
    
    # Diagonal matrix
    # diangonal entries are (#triangle)*1/3 of triangle
    # #triangle  = the number of triangle which consists of the vertex i

    VA = sparse.spdiags(vertex_area, 0, N, N)
    return L, VA

class GeodesicDistHeatMethod(object):
    """
    Computation of geodesic distances on triangle meshes using the heat method
    """
    def __init__(self, verts, tris, m = 1e2):
        """
        Arguments
            verts: ndarray, shape=[#vert, 3]
            tris: ndarray, shape=[#face, 3]
        """
        self._verts = verts
        self._tris = tris
        self.N = len(verts)
        # cyclic order: clock cycle
        # precomputation 
        e01 = verts[tris[:, 1]] - verts[tris[:, 0]]
        e12 = verts[tris[:, 2]] - verts[tris[:, 1]]
        e20 = verts[tris[:, 0]] - verts[tris[:, 2]]

        # normalized edges
        Ne01 = normalized(e01)
        Ne12 = normalized(e12)
        Ne20 = normalized(e20)

        # triangle area 
        self._triangle_area = 0.5 * vecnorm(np.cross(e01, e12))

        unit_normal = normalized(np.cross(normalized(e01), normalized(e12)))

        self._unit_normal_cross_e01 = np.cross(unit_normal, e01)
        self._unit_normal_cross_e12 = np.cross(unit_normal, e12)
        self._unit_normal_cross_e20 = np.cross(unit_normal, e20)

        # parameters for heat method
        h = np.mean([vecnorm(e01), vecnorm(e12), vecnorm(e20)])
        t = m * h ** 2

        # pre-facctorize possion system
        Lc, A = compute_topological_laplacian(verts, tris)
        B = (A-t*Lc)
        self._factored_AtLc = splu(B.tocsc())
        self._factored_Lc = splu(Lc.tocsc())
        self.phi = None

    def __call__(self, idx):
        """
        computes geodesic distance to all vertices in the mesh
        idx can be either an integer (single vertex index) or a list of vertex indices
        or an array of bools of length N
        """

        u_0 = np.zeros_like(self._verts)
        u_0[idx] = 1

        # heat flow u_t (time t)
        u_t = self._factored_AtLc.solve(u_0)

        # gradient of u_t
        grad_u = (1 / (2 * self._triangle_area)[:, None]) * (
            self._unit_normal_cross_e01 * u_t[self._tris[:, 2]]
            + self._unit_normal_cross_e12 * u_t[self._tris[:, 0]]
            + self._unit_normal_cross_e20 * u_t[self._tris[:, 1]]
        )

        # gradient field X
        X = -grad_u / vecnorm(grad_u)[:, None]

        div_Xs = np.zeros(self.N)

        # compute divergence of X 
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            vi1, vi2, vi3 = self._tris[:, i1], self._tris[:, i2], self._tris[:, i3]
            e1 = self._verts[vi2] - self._verts[vi1]
            e2 = self._verts[vi3] - self._verts[vi1]
            e_oop = self._verts[vi3] - self._verts[vi2]
            cot1 = 1/np.tan(np.arccos((normalized(-e2) * normalized(-e_oop)).sum(axis = 1))) #e2_norm^T*e_oop = cos(theta)
            cot2 = 1/np.tan(np.arccos((normalized(-e1) * normalized(e_oop)).sum(axis = 1)))
            div_Xs += np.bincount(vi1.astype(int), 0.5 * (cot1*(e1*X).sum(axis = 1) + cot2 * (e2*X).sum(axis = 1)), minlength = self.N)
        
        # distance function
        phi = self._factored_Lc.solve(div_Xs)
        phi -= phi.min()
        self.phi = phi

        return phi
    
    def visualize_distance_func(self, cmap = 'jet'):
        VisPointsAttributes(self._verts, self.phi, cmap)
            
            

       


