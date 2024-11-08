import numpy as np

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

a = np.array([np.nan, np.nan, 10, np.nan, 1, np.nan, np.nan])
nans, x = nan_helper(a)
a[nans] = np.interp(x(nans), x(~nans), a[~nans])
print(a)

# two-dimensional case
def nan_filler(y: np.ndarray) -> np.ndarray:
    """Fill nans along 1st dimension.

    y: (n_steps, ...)
    """
    shape = y.shape
    y = y.reshape((shape[0], -1)).transpose()  # (emb, n_steps)
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    y = y.transpose().reshape(shape) # original shape
    return y
 


a = np.array([[np.nan, 5], [2, np.nan]])
print(nan_filler(a))


