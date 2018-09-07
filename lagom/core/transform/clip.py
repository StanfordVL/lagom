import numpy as np

from .base_transform import BaseTransform


class Clip(BaseTransform):
    r"""Clip the input data"""
    def __call__(self, x, a_min, a_max):
        r"""Clip values by min/max bounds
        
        Args:
            x (object): input data
            a_min (float): minimum value
            a_max (float): maximum value
        
        Returns
        -------
        out : ndarray
            clipped data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        
        # Clip the data
        out = np.clip(x, a_min=a_min, a_max=a_max)
        
        return out.astype(np.float32).tolist()  # enforce raw float dtype
