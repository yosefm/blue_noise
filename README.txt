
Mitsa-Parker blue noise generator
=================================

This repository is my ongoing lab for a Python implementation of an algorithm
for generating blue noise masks for dithering, first published by T. Mitsa and 
K. Parker, in 1992. See bibliographic entry at the top of mitsa_parker.py.

The main reason for using this algorithm over the more widely-known void-
and-cluster method is its speed. 

As lab code, I give no guaranties. Especially, some runs may randomly fail 
because of some edge case I haven't handled. If you're looking for a quick
copy-paste job, you're out of luck. You need to know what's going on to use it.

That being said, here's how to use it:
1. Get prerequisites
2. build the support module with Cython.
3. Run with Python 3 as detailed below.


Prerequisites
-------------
The following packages are used, besides Python 3:
* SciPy (at least 1.5.2)
* Matplotlib (at least 3.3.0)
* Cython (at least 0.29.21)


Building
--------
Some core functionality is handled with Cython. the algorithm is pretty fast
even without it, but I was in the mood. If you want a pure-python version,
look through the few commits, it's there. Otherwise, after Cython is installed, 
run:

  $ python3 setup.py build_ext --inplace

Note that the `python3` command may be calld just `python` on your system. 
The above line is how it is in Ubuntu Linux, for now.


Running
-------
After the Cython module is built, go for this:

  $ python3 mitsa_parker.py --help

It'll tell you what to do and detail some features (see below). 

Real example: to create a 2D mask of side 256, and view the results, run
    
  $ python3 mitsa_parker.py --view
  
(no need to give arguments because those are the defaults.)


Features
--------
The default run creates a 256x256 2D blue-noise mask. It is saved an an `.npy` 
file for loading into other scripts. th `--view` option shows you the spectrum
and a number of grey-level thresholds of the blue noise mask. But that's not all.

* Dimensions - the script can create 3D masks (or, in fact, any dimnsion, though 
  I only tested with 2 and 3).

* Anisometry - the spectrum can change, being wider for one dimension than 
  another. this handles cases where the underlying resolution of the device
  on which you apply the mask is anisometric. See the command-line switch
  `--squeeze-factors`
  
That's it for now. Have fun.


Contact
-------
Written by Yosef Meller <yosefm@gmail.com>
