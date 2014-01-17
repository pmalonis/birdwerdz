#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from distutils.command.build_ext import build_ext
import numpy

setup(
    name = "birdwerdz",
    cmdclass = {"build_ext" : build_ext},
    ext_modules = [Extension("birdwerdz",
                             sources=["birdwerdz/birdwerdz.pyx", "birdwerdz/c_functions.c"],
                             include_dirs=['.', numpy.get_include()])],
    scripts = ['classify.py', 'plot_cluster_means.py', 'select_cluster.py', 'labeling.py', 'cluster.py']
)

