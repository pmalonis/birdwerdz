#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from distutils.command.build_ext import build_ext
import numpy

setup(
    name = "birdwerdz",
    entry_points = {'console_scripts':'birdwerdz=birdwerdz._main:main'},
    packages = ['birdwerdz'],
    cmdclass = {"build_ext" : build_ext},
    ext_modules = [Extension("birdwerdz.dtw",
                             sources=["birdwerdz/dtw.pyx", "birdwerdz/c_functions.c"],
                             include_dirs=['.', numpy.get_include()])],
)


