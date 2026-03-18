from setuptools import setup
from distutils.extension import Extension
from os import path
import glob

from Cython.Build import cythonize
import numpy as np

include_dirs = [np.get_include()]

scripts = glob.glob('bin/*')

this_directory = path.abspath(path.dirname(__file__))
# noinspection PyPackageRequirements
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

extensions = [Extension('maipc.DBN.viterbi', ['maipc/DBN/viterbi.pyx'],
                        include_dirs=include_dirs)]

requirements = ['numpy>=1.19.4',
                'scipy>=0.16',
                'cython>=0.25',
                'mido>=1.2.6',
                'madmom>=0.16.1',
                'KDEpy>=1.0.10',
                'scipy>=1.5.4',
                'fire>=0.4.0',
                ]

setup(name='maipc',
      version='0.1',
      packages=['maipc'],
      ext_modules=cythonize(extensions),
      install_requires=requirements,
      scripts=scripts,
      description='Pulse clarity metrics based on the madmom package',
      long_description=long_description,
      author='Nicolás Pironio',
      )
