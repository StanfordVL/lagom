import numpy as np
from sklearn.metrics import explained_variance_score

from .base_transform import BaseTransform


class ExplainedVariance(BaseTransform):
    r"""Computes the explained variance regression score.
   
    It involves a fraction of variance that the prediction explains about the ground truth.
   
    Let :math:`\hat{y}` be the predicted output and let :math:`y` be the ground truth output. Then the explained
    variance is estimated as follows:
   
    .. math::
        \text{EV}(y, \hat{y}) = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
   
    The best score is :math:`1.0`, and lower values are worse. A detailed interpretation is as following:
   
    * :math:`\text{EV} = 1`: perfect prediction
    * :math:`\text{EV} = 0`: might as well have predicted zero
    * :math:`\text{EV} < 0`: worse than just predicting zero
   
    .. note::
    
        It calls the function from ``scikit-learn`` which handles exceptions better e.g. zero division..
        
    Example::
    
        >>> f = ExplainedVariance()
        >>> f(y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0.0, 2, 8])
        0.9571734666824341
        
        >>> f(y_true=[[0.5, 1], [-1, 1], [7, -6]], y_pred=[[0, 2], [-1, 2], [8, -5]])
        0.9838709533214569
        
        >>> f(y_true=[[0.5, 1], [-1, 10], [7, -6]], y_pred=[[0, 2], [-1, 0.00005], [8, -5]])
        0.6704022586345673
   
    """
    def __call__(self, y_true, y_pred, **kwargs):
        r"""Estimate the explained variance.
       
        Args:
            y_true (object): ground truth output
            y_pred (object): predicted output
            **kwargs: keyword arguments to specify the estimation of the explained variance. 
           
        Returns
        -------
        out : float
            estimated explained variance
        """
        assert not np.isscalar(y_true), 'does not support scalar value !'
        assert not np.isscalar(y_pred), 'does not support scalar value !'

        y_true = self.to_numpy(y_true, np.float32)
        y_pred = self.to_numpy(y_pred, np.float32)
       
        ev = explained_variance_score(y_true=y_true, y_pred=y_pred, **kwargs).astype(np.float32)
       
        return ev
