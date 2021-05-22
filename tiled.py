# -*- coding: utf-8 -*-
"""
Explore the tileability and repetition patterns of the blue-noise masks.

Created on Fri Apr 10 11:03:06 2020

@author: Yosef.Meller
"""

if __name__ == "__main__":
    import numpy as np
    from scipy import signal as sig
    from matplotlib import pyplot as pl
    
    maskname="exp/bn_255x255.npy"
    tile = np.load(maskname)
    
    # the tiled version:
    mosaic = np.tile(tile, (5, 5))
    
    pl.imshow(mosaic)
    pl.title("Mosaic")
    pl.colorbar()
    
    # Thresholded:
    thrs = mosaic < 64
    pl.figure()
    pl.imshow(thrs)
    pl.title("Thresholded")
    #pl.colorbar()
    
    # Moving average:
    mavg = sig.convolve2d(mosaic, np.ones((3, 3)))
    
    pl.figure()
    pl.imshow(mavg)
    pl.title("Local average")
    pl.colorbar()
    
    pl.show()
    