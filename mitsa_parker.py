#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a blue-noise mask using the Mitsa-Parker method.

References:
[1] Mitsa, T. and Parker, K.J. (1992). Digital halftoning technique using a
    blue noise mask. Opt. Soc. Am. A, vol 8.
[2] Smith, S.W. (1997-1998), The Scientist and Engineer's Guide to Digital 
    Signal Processing. Available online: dspguide.com 

Created on Mon Mar 30 11:19:33 2020

@author: Yosef.Meller
"""

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as pl

def desired_PSD_nd(grey_level, side_length, num_dims=2):
    """
    Generates a radially-averaged PSD for a 2D square mask.
    
    Arguments:
    grey_level - [0,1]
    side_length - number of samples in signal. The result PSD is half that length.
    """
    # It may seem strange that the power is the square of energy, the units
    # don't seem to add up. But it's ok. Energy is the sum of sample squares,
    # and power is the square of sample sum. But since all samples are 0/1,
    # the square is not performed for the energy calculation, it's not needed.
    num_samples = side_length**num_dims
    total_energy = num_samples * grey_level
    DC_power = total_energy**2
    
    filterRanges = [np.s_[-sz//2 : sz//2] for sz in (side_length,)*num_dims]
    coords = np.mgrid[filterRanges] # n x shape
    
    # This gives the shape but not the correct strength:
    cf = np.sqrt(grey_level)
    r = np.linalg.norm(coords, axis=0) # r.shape == `shape`
    peak = cf*r.max()
    flt = np.sin(0.5*np.pi*r/peak)**4
    post_peak = 1.1*peak
    flt[r > post_peak] = np.sin(0.5*np.pi*post_peak/peak)**4

    #print(flt.sum())
    
    # Parseval's relation, see [2] ch. 10 p. 208:
    # total_energy = PSD.sum()/N
    # Decompose right hand into available and wanted parts:
    # total_energy = (DC**2 + required.sum())/N
    # N*total_energy = DC**2 + required.sum()
    # required.sum() = N*total_energy - DC**2
    required_sum = num_samples*total_energy - DC_power
    flt *= required_sum/flt.sum()
    
    #print("Required:", total_energy, (DC_power + flt.sum())/num_samples)
    
    flt[(side_length//2,)*num_dims] = DC_power
    return flt


def radially_average(spectrum):
    """
    Turn an n-d array into a 1-d array where each cell is the average of all 
    input cells at a radial bin. Radius is measured from array center.
    """
    num_bins = int(np.ceil(np.linalg.norm(spectrum.shape)/2))
    ret = np.empty(num_bins)
    r = np.linalg.norm(
        np.mgrid[[np.s_[-sz//2 : sz//2] for sz in spectrum.shape]],
        axis=0
    )
    
    for bin_ix in range(num_bins):
        annulus = spectrum[(r >= bin_ix) & (r < bin_ix + 1)]
        ret[bin_ix] = annulus.sum() / len(annulus)
    
    return ret


def radially_distribute(spectrum, shape):
    """
    The opposite of `radially_averaged_spectrum()`. Turns an 1-d array into 
    an n-d array, where each cell i,j,k gets the value in the radial bin 
    corresponding to the distance of point 9i,j,k) from the n-d array's center.
    """
    ret = np.empty(shape)
    r = np.linalg.norm(
        np.mgrid[[np.s_[-sz//2 : sz//2] for sz in shape]],
        axis=0
    )
    
    spect_sqrt = np.sqrt(spectrum)
    for bin_ix in range(len(spectrum)):
        ret[(r >= bin_ix) & (r < bin_ix + 1)] = spect_sqrt[bin_ix]
    
    return ret


def correct_signal(signal, desired_PSD):
    """
    Given a binary signal, edit its radial power spectral density to match the
    desired PSD, then create a non-binary signal from the original one, using
    the corrected PSD.
    
    Arguments:
    signal - 2-d array representing the time-domain signal.
    desired_PSD - 1-d array, number of cells is ceil(half the diagonal of 
        `signal`)
    
    Returns:
    2-d array, same shape as `signal`.
    """
    sig_spect = np.fft.fftshift(np.fft.fftn(signal))
    sig_PSD_ra = radially_average(sig_spect*sig_spect.conj())
    
    ratio = np.sqrt(desired_PSD/sig_PSD_ra)
    ratio_filt = radially_distribute(ratio, signal.shape)
    
    corrected_spect = sig_spect*ratio_filt
    corrected_sig = np.real(np.fft.ifftn(np.fft.ifftshift(corrected_spect)))
    # The imaginary part should be ~zero, np.real() just squeezes it out.
    
#    # diagnostics:
#    energy = np.sum(corrected_sig**2)
#    print("Energy:", energy)
    
    return corrected_sig


def iterate_signal(signal, desired):
    """
    Does one iteration of the Mitsa-Parker PIPPSMA algorithm.
    
    Arguments:
    signal - an (n,n) boolean array, the current state.
    desired - the radially-averaged PSD of the *target* signal.
    
    Returns:
    a new signal based on the input signal with replacements according to 
    BIPPSMA [1].
    """
    corrected_sig = correct_signal(signal, desired)
    
    # Do some replacements:
    num_replacements = signal.shape[0] # assumes square/cube array. revisit later.
    error = np.abs(corrected_sig - signal)
    
    # Identify worst pairs.
    void = signal == 0
    void_error = np.where(void, error, 0)
    stuff_error = np.where(~void, error, 0)
    void_error_order = np.argsort(-void_error, None)[:num_replacements] # descending.
    stuff_error_order = np.argsort(-stuff_error, None)[:num_replacements] # descending.
    
    new_sig = signal.copy().flatten()
    np.put(new_sig, void_error_order, 1)
    np.put(new_sig, stuff_error_order, 0)
    
    new_sig = new_sig.reshape(*signal.shape)
    mse = np.nansum((new_sig - corrected_sig)**2)
    return new_sig, mse
    

def initial_binary_mask(grey_level, side_length, num_dims=2):
    """
    Prepares an initial binary mask for the ACBNOM, using the BIPPSMA stage. [1]
    Note some magic numbers related to the iteration control. Revisit later.
    """
    num_samples = side_length**num_dims
    
    # white noise:
    # Instead of thresholding a uniform distribution, I make sure the number
    # of 1s is exactly as required by the grey level (up to discretization...)
    perm = np.random.permutation(np.r_[:num_samples])
    signal = np.zeros(num_samples)
    num_on = int(num_samples*grey_level)
    signal[perm[:num_on]] = 1
    signal = signal.reshape((side_length,)*num_dims)
    
    # Generate the radial PSD that is the target function for iteration.
    actual_grey_level = signal.mean()
    desired = desired_PSD_nd(actual_grey_level, side_length, num_dims)
    desired_radial = radially_average(desired)
    
    new_sig = np.zeros_like(signal)
    new_sig_last = signal
    mse = 1000000000
    i = 0
    while i < 100 and not np.isnan(mse):
        new_sig, new_mse = iterate_signal(new_sig_last, desired_radial)
        #print(i, new_mse)
        if (new_mse > mse and i > 10):
            break
        mse = new_mse
        new_sig_last = new_sig
        i += 1
    
    return new_sig, desired_radial


def iterate_grey_level(prev_mask, new_g_disc, num_grey_levels=256, upward=True):
    """
    Create a new binary mask for level $g + \Delta g$. This is the main part
    of an ACBNOM stage. See [1]
    
    Arguments:
    prev_mask - the mask for the grey level that is one discretization step
        below the current level. E.g. if the current discretized level is 
        128, then the previous mask belongs to level 127, or ~0.5 in 0..1 
        scale.
    new_g_disc - the discretized grey level, where 1 -> num_grey_levels - 1, 
        e.g for 8-bit greyscale, 1 -> 255 and 0.5 -> 127.
    num_grey_levels - number of possible levels, e.g. 256 for 8-bit greyscale.
    upward - direction of construction. Affects the kind of replacements made.
    """
    gl_delta = 1./num_grey_levels
    grey_level = new_g_disc/(num_grey_levels - 1)
    
    # Create desired spectrum.
    desired = desired_PSD_nd(
        new_g_disc*gl_delta, prev_mask.shape[0], prev_mask.ndim)
    desired_radial = radially_average(desired)
    
    # Find error:
    corrected_sig = correct_signal(prev_mask, desired_radial)
    error = np.abs(corrected_sig - prev_mask)
    
    # Make corrections:
    num_replacements = int(np.multiply.reduce(prev_mask.shape)*gl_delta)
    
    ## Identify worst zeros. This is different than BIPPSMA, because we 
    ## have to check each replacement's neighbourhood to avoid clusters.
    if upward:
        replace_value = 0
        replace_to = 1
    else:
        replace_value = 1
        replace_to = 0
    
    void = prev_mask == replace_value
    void_error = np.where(void, error, 0)
    void_error_order = np.argsort(-void_error, None)# descending.
    
    ## Replace:
    new_sig = prev_mask.copy()
    error_coords = np.unravel_index(void_error_order[:void.sum()], prev_mask.shape)
    
    # We need to make sure replacements don't cluster, by observing the local
    # means. We do that for the entire array - in NumPy. It's cheaper than
    # doing it individually per point in pure Python.
    half_window = 4
    window_size = (2*half_window + 1)
    window = np.full((window_size,)*prev_mask.ndim, 1/window_size**prev_mask.ndim)
    local_mean = ndi.convolve(prev_mask, window, mode='wrap')
    
    for coords in zip(*error_coords):
        if upward:
            crowded = local_mean[coords] > grey_level
        else:
            crowded = local_mean[coords] < grey_level
            
        if crowded:
            continue
        
        assert(new_sig[coords] == replace_value)
        new_sig[coords] = replace_to
        num_replacements -= 1
        if num_replacements == 0:
            break
    
    # Profit:
    return new_sig

def gen_blue_noise(side_length, num_dims=2):
    """
    Create an array whose pixels are numbered 0..array size. Thresholding the
    array at any level gives a binary mask that is distributed as blue noise 
    with density matching the threshold grey level. Algorithm: [1].
    """
    grey_level = 0.5
    num_grey_bits = 8
    num_grey_levels = 2**num_grey_bits
    
    init_mask, desired_radial = initial_binary_mask(grey_level, side_length)
    
    # The cumulative array:
    numbered_pixels = np.where(init_mask, (num_grey_levels >> 1) - 1, num_grey_levels >> 1)
         
    next_mask = init_mask.copy()
    for grey_level_disc in range((num_grey_levels >> 1) + 1, num_grey_levels):
        print(grey_level_disc)
        next_mask = iterate_grey_level(next_mask, grey_level_disc)
        numbered_pixels[next_mask == 0] += 1
    
    next_mask = init_mask.copy()
    for grey_level_disc in range((num_grey_levels >> 1) - 1, 0, -1):
        print(grey_level_disc)
        next_mask = iterate_grey_level(next_mask, grey_level_disc, upward=False)
        numbered_pixels[next_mask != 0] -= 1
   
    return numbered_pixels
    

if __name__ == "__main__":
    side_length = 256
    numbered_pixels = gen_blue_noise(side_length)
    
    # Examine result:
    pl.figure()
    pl.imshow(numbered_pixels)
    pl.colorbar()
    
    pl.figure()
    next_FT = np.fft.fftshift(np.fft.fft2(numbered_pixels))
    next_PSD = next_FT*next_FT.conj()
    pl.plot(radially_average(next_PSD)[1:])
    
    #"""
    # A few slices, self check:
    for slice_num in [10, 50, 128, 170, 200, 255]:
        bin_mask = numbered_pixels <= slice_num
        pl.figure()
        pl.imshow(bin_mask)
        pl.title("Slice %d" % slice_num)
    #"""
    
    pl.show()
    
