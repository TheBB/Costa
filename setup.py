#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='Costa',
    version='0.1.0',
    description='Corrective Source Term Approach',
    url='https://github.com/TheBB/Costa',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    license='GNU public license v3',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
