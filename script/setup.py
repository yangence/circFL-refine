#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from circFL_refine.version import __version__

setup(name='circFL-refine',
    version=__version__,
    description='circFL-refine: a tool for full-length circRNA isoform construction based on convolutional neural network',
    author='Zelin Liu',
    author_email='zlliu@bjmu.edu.cn',
    url='https://github.com/yangence/circFL-refine',
    license='GPL3',
    keywords='circular RNAs',
    python_requires=">=3",
    packages=find_packages(),
    install_requires=[
        'pandas>=0.25.2',
        'numpy>=1.17.5',
        'docopt>=0.6.2',
        'scikit-learn>=0.24.2',
        'pyfasta>=0.5.2',
        'torch>=1.7.1'
    ],
    entry_points={
      'console_scripts': [
          'circFL-refine=circFL_refine.circFL_refine_main:main',
      ],
    }
)
