# -*- coding: utf-8 -*-
"""
Generate a 3D blue-noise slab and explore its properties.

Created on Wed Mar 11 09:22:09 2020

@author: Yosef.Meller
"""

import numpy as np, matplotlib.pyplot as pl
#from blue_noise_gen import create_dither_array, get_clean_spectrum

def get_clean_spectrum(signal, thresh=1000):
    spect = np.abs(np.fft.fftn(signal))
    
    # remove DC:
    spect[0,0] = 0
    
    return np.fft.fftshift(spect)


def display_3d(viewable, title=""):
    """
    Produce a plot with subplots spreading out slices of `viewable`.
    """
    pl.figure()
    pl.suptitle(title)
    
    horiz_subplt_num = int(np.sqrt(len(viewable)))
    vert_subplt_num = int(np.ceil(len(viewable) / horiz_subplt_num))
    
    pl.subplot(horiz_subplt_num, vert_subplt_num, 1)
    setsub = lambda index: pl.subplot(vert_subplt_num, horiz_subplt_num, index + 1)
    
    for sliceIx, slc in enumerate(viewable):
        setsub(sliceIx)
        pl.axis('off')
        pl.imshow(slc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser();
    parser.add_argument("--size", type=int, default=32, 
        help="Output dither array have cube dimentions SIZE**3.")
    parser.add_argument("--compress-z-hole", '-c', type=float, default=1,
        help="Change the filters such that the 0-energy part of the 3D "
        "spectrum will be divided by this on the z axis.")
    
    whatdo = parser.add_mutually_exclusive_group()
    whatdo.add_argument("--view", 
        help="View a previously generated slab from this file.")
    whatdo.add_argument("--output", "-o", help="Save 3D slab to this name.")
    args = parser.parse_args();
    
    if args.view is not None:
        slab = np.load(args.view)
    else:
        slab = create_dither_array(
            (args.size,)*3, np.r_[args.compress_z_hole, 2, 1])
    
    if args.output is None:
        spectrum = get_clean_spectrum(slab)
        display_3d(spectrum, title="3D spectrum")
        
        # The horizontal 2D spectra, for comparison:
        spectra_2d = [get_clean_spectrum(slab[s], 450) for s in range(len(spectrum))]
        display_3d(spectra_2d, title="2D spectra - horizontal")
        
        spectra_2d = [get_clean_spectrum(slab[:,s], 450) for s in range(len(spectrum))]
        display_3d(spectra_2d, title="2D spectra - vertical")
        
        pl.show()
        
    else:
        np.save(args.output, slab)
    