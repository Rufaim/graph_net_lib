from setuptools import find_packages
from setuptools import setup


description = """Graph Nets is a library for building graph networks in Tensorflow 2.x.
See paper "Relational inductive biases, deep learning, and graph networks" for details.
"""

setup(
    name="graphnetlib",
    version="1.0.0",
    description="Library for building graph networks using Tensorflow 2.x.",
    long_description=description,
    author="Rufaim",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "tensorflow", "keras", "machine learning"],
    url="https://github.com/Rufaim/graph_net_lib",
    packages=find_packages(),
    install_requires=[
        "tensorflow>2.2.0",
        "networkx",
        "numpy<1.20",
        "setuptools",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)