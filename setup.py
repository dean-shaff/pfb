#!/usr/bin/env python

from distutils.core import setup

from src.pfb import __version__

setup(
    name="pfb",
    version=__version__,
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
