from dataclasses import dataclass
import numpy as np

@dataclass
class GaussianProp:
        xyz: np.ndarray
        normals: np.ndarray
        f_dc: np.ndarray #this is SH_coeffs, needs to be converted to RGB by SH2RGB
        f_rest: np.ndarray
        opacities: np.ndarray
        scale: np.ndarray
        rotation: np.ndarray
        covariance: np.ndarray