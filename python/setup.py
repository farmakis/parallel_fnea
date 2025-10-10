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
import os.path as osp
import re


########################################################################
#                     Targets and compile options                      #
########################################################################

###  targets and compile options  ###
to_compile = [ # comment undesired extension modules
    "fnea_partition_cpy",
]

# Compilation and linkage options
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
# COMP_T_ON_32_BITS for components identifiers on 32 bits rather than 16
if os.name == 'nt':  # windows
    extra_compile_args = ["/std:c++11", "/openmp",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix':  # linux
    extra_compile_args = ["-std=c++11", "-fopenmp", "-g", "-O0",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["-lgomp", "-g"]
else:
    raise NotImplementedError('OS not supported yet.')


########################################################################
#                         Auxiliary functions                          #
########################################################################

class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin"

    def run(self):
        build_path = self.build_lib


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(osp.join(dir, f))


########################################################################
#                           Parallel FNEA                              #
########################################################################

###  preprocessing  ###

# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

if not osp.exists("bin"):
    os.mkdir("bin")

if not os.path.exists("./bin/"):
    os.mkdir("bin/")

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

# Postprocessing
try:
    # remove temporary compilation products
    shutil.rmtree("build")
except FileNotFoundError:
    pass

# Restore the initial working directory
os.chdir(tmp_work_dir)           
