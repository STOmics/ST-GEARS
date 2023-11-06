#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/06/23 11:41 PM
# @Author  : Tianyi Xia
# @File    : setup.py
# @Email   : xiatianyi@genomics.cn

import setuptools

__version__ = "1.0.0"

requirements = open("requirements.txt").readline()


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="st_gears",
    version=__version__,
    author="Tianyi Xia",
    author_email="xiatianyi@genomics.cn",
    description="A Spatial Transcriptomics Geospatial Profile Recovery Tool through Anchors",
    long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/STOmics/ST-GEARS.git",
    packages=setuptools.find_packages(),
    package_data={'': ["*.so"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
)