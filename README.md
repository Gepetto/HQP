# HQP

Library for hierarchichal quadratic programming

## Characteristics:

- Computes HQP based on nulsspace projections and quadritc programming using qpOASES and scipy.
- It can be used for generation motion of anthromorphic systems such as robots and avatars.

## Setup:
```
$ python setup.py --prefix='your install path'
```

## Required Dependencies:

- Python 2.7, numpy, scipy, matplotlib:
```
$ sudo apt-get install python-numpy python-scipy python-matplotlib
```
- Cython: c-extensions for python
```
$ sudo pip install cython
```
- robotpkg: http://robotpkg.openrobots.org/debian.html ; it will allow you to get:
    - qpOASES:
    - Pinocchio: library for rigid multi-body dynamics.
    - Gepetto-viewer: A graphical interface for pinocchio.
    - Gepetto-viewer-corba: CORBA server/client for the Graphical Interface of Pinocchio.
```
$ sudo apt-get install robotpkg-py27-pinocchio robotpkg-qpoases+doc robotpkg-gepetto-viewer-corba
```

## Extra Dependencies:

- Models: Contains biomechanical and robotic models. Can be downloaded from here: https://github.com/GaloMALDONADO/Models
- gmp: free library for arbitrary precision arithmetic, operating on signed integers, rational numbers, and floating-point numbers.
```
$ sudo apt-get install  libgmp3-dev
```
- pycddlib: the simplest way to install is with pip:
```
$ pip install pycddlib
```
