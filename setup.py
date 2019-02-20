#!/usr/bin/env python

from distutils.core import setup

setup(
    name="pfb",
    version="0.1.0",
    description="PFB channelization and inversion",
    author="Dean Shaff",
    author_email="dean.shaff@gmail.com",
    url="https://github.com/dean-shaff/pfb-channelizer",
    packages=["pfb"],
    package_dir={"pfb": "src/pfb"},
    requires=[
        "numpy",
        "scipy",
        "numba",
        "llvmlite"
    ]
)
