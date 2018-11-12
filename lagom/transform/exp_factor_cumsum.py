import numpy as np

from scipy.signal import lfilter

from .base_transform import BaseTransform


class ExpFactorCumSum(BaseTransform):
    r"""Calculate future accumulated sums for each element in a list with an exponential factor. 
    
    Given input data :math:`x_1, \dots, x_n` and exponential factor :math:`\alpha\in [0, 1]`, it returns
    an array :math:`y` with the same length and each element is calculated as following
    
    .. math::
        y_i = x_i + \alpha x_{i+1} + \alpha^2 x_{i+2} + \dots + \alpha^{n-i-1}x_{n-1} + \alpha^{n-i}x_{n}
            
    .. note::
        To gain the optimal runtime speed, we use ``scipy.signal.lfilter`` for non-mask version.
        And when we use binary masks, we use a vectorized implementation. 
    
    Example::
    
        >>> f = ExpFactorCumSum(0.1)
        >>> f([1, 2, 3, 4])
        array([[1.234, 2.34 , 3.4  , 4.   ]])
        
        >>> f([1, 2, 3, 4], mask=[1, 0, 1, 1])
        array([[1.2, 2. , 3.4, 4. ]], dtype=float32)
        
        >>> f([[1, 2, 3, 4], [5, 6, 7, 8]], mask=[[1, 0, 1, 1], [1, 1, 0, 1]])
        array([[1.2 , 2.  , 3.4 , 4.  ],
               [5.67, 6.7 , 7.  , 8.  ]], dtype=float32)

    """
    def __init__(self, alpha):
        r"""Initialize transformation. 
        
        Args:
            alpha (float): exponential factor between zero and one. 
        """
        self.alpha = alpha
        
    def __call__(self, x, mask=None):
        r"""Calculate future accumulated sums with exponential factor. 
        
        An optional binary mask could be used. 
        Intuitively, the computation will restart for each occurrence
        of zero in the mask. If ``None``, then default mask is ones everywhere.
        
        Args:
            x (list): input data
            mask (list, optional): binary mask (zero or one) corresponds to each element. 
                Default: ``None``
            _fast_code (bool, optinal): if ``True``, then use fast implementation based 
                on build-in function ``accumulate()``. Otherwise, use vanilla implementation.
                Default: ``True``

        Returns
        -------
        out : ndarray
            calculated data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        x = self.to_numpy(x, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        
        if mask is not None:
            assert np.asarray(mask).dtype != np.bool
            assert np.array_equal(mask, np.array(mask).astype(bool))
            
            mask = self.to_numpy(mask, np.float32)
            if mask.ndim == 1:
                mask = np.expand_dims(mask, 0)
            assert mask.ndim == 2
            assert mask.shape == x.shape
        
        if mask is None:
            return lfilter([1], [1, -self.alpha], x[:, ::-1], axis=1)[:, ::-1]
        else:
            N, T = x.shape
            out = np.zeros_like(x, dtype=np.float32)
            cumsum = np.zeros(N, dtype=np.float32)

            for t in reversed(range(T)):
                cumsum = x[:, t] + self.alpha*cumsum*mask[:, t]
                out[:, t] = cumsum

            return out
