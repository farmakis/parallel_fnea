#!/usr/bin/env python3
"""
Setup script for compiling parallel FNEA python extensions

Compilation command: python setup.py build_ext

"""

from setuptools import setup, Extension
from distutils.command.build import build
import numpy
import shutil
import os 
import re

###  targets and compile options  ###
to_compile = [
    "fnea_partition_cpy"
]

# compilation and linkage options
# _GLIBCXX_PARALLEL is only useful for libstdc++ users
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if os.name == 'nt': # windows
    extra_compile_args = [
        "/std:c++11", "/openmp", "-D_GLIBCXX_PARALLEL",
        "-DMIN_OPS_PER_THREAD=10000",
    ]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix': # linux
    extra_compile_args = [
        "-std=c++11", "-fopenmp", "-D_GLIBCXX_PARALLEL",
        "-DMIN_OPS_PER_THREAD=10000", "-O3", "-march=native"
    ]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')

###  auxiliary functions  ###

class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin" 
    def run(self):
        build_path = self.build_lib

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

###  preprocessing  ###

# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

if not os.path.exists("./bin/"):
    os.mkdir("bin")

# remove previously compiled lib
for shared_obj in to_compile: 
    purge("bin/", shared_obj)

###  compilation  ###

name = "fnea_partition_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/fnea_partition_cpy.cpp", "../src/fnea_partition.cpp"],
            include_dirs=[numpy.get_include(), # find the Numpy headers
                "../include"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

###  postprocessing  ###
try:
    shutil.rmtree("build") # remove temporary compilation products
except FileNotFoundError:
    pass

os.chdir(tmp_work_dir) # get back to initial working directory