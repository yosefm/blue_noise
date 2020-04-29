"""
Local average in given coordinates.
Created on Tue Apr 28 12:24:32 2020

@author: yosef
"""

import numpy as np
cimport numpy as np
cimport cython

cdef class DimentionalConverter:
    cdef np.ndarray annuli
    cdef np.ndarray annuli_weight
    cdef unsigned int num_bins
    cdef tuple _shape
    
    def __init__(self, shape, squeeze_factors=None):
        """
        shape - of the nD side of the conversion. A tuple.
        squeeze_factors - n-length array. Change the calculation of radius
            to be ellipsoidal, r = sqrt(sum(coord_i**2/squeeze_factor_i**2))
        """
        self._shape = shape
        self.num_bins = int(np.ceil(np.linalg.norm(shape)/2))
        
        if squeeze_factors is None:
            squeeze_factors = np.ones(len(shape))
        
        # Make squeeze-factors broadcastable to coords array shape:
        squeeze_factors_shape = (len(squeeze_factors),) + (1,)*len(shape)
        squeeze_factors = squeeze_factors.reshape(*squeeze_factors_shape)
        
        coords = np.mgrid[[np.s_[-sz//2 : sz//2] for sz in shape]]
        r = np.linalg.norm(coords / squeeze_factors, axis=0)
        annuli = np.digitize(r, bins=np.arange(1, self.num_bins + 1))
        self.annuli = annuli.flatten()
        self.annuli_weight = np.bincount(self.annuli)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def radially_average(self, np.ndarray spectrum):
        """
        Turn an n-d array into a 1-d array where each cell is the average of all 
        input cells at a radial bin. Radius is measured from array center.
        
        Arguments:
        spectrum - array to convert, same shape as given to constructor.
        """
        cdef:
            np.ndarray[np.float64_t, ndim=1] source = spectrum.flatten()
            np.ndarray[np.float64_t, ndim=1] ret = np.zeros(self.num_bins)
            np.ndarray[np.int_t, ndim=1] annuli = self.annuli
            unsigned int ix
        
        for ix in range(source.size):
            ret[annuli[ix]] += source[ix]
        
        return ret/self.annuli_weight
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def radially_distribute(self, np.ndarray[np.float64_t, ndim=1] spectrum):
        """
        The opposite of `radially_averaged_spectrum()`. Turns a 1-d array into 
        an n-d array, where each cell i,j,k gets the value in the radial bin 
        corresponding to the distance of point i,j,k) from the n-d array's center.
        """
        cdef:
            np.ndarray[np.int_t, ndim=1] annuli = self.annuli
            np.ndarray[np.float64_t, ndim=1] ret = np.empty(self._shape).flatten()
            unsigned int ix
        
        for ix in range(ret.size):
            ret[ix] = spectrum[annuli[ix]]
        
        return ret.reshape(*self._shape)
