import setuptools
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dinos",
    version="0.1.1",
    author="Alexandre Manoury",
    author_email="alex@pika.tf",
    description="Active Learning Environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pikalchemist/dinos",
    ext_modules=cythonize("dinos/data/operations.pyx"),
    include_dirs=[np.get_include()],
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["exlab", "Cython", "matplotlib==3.2", "numpy", "panda3d",
                      "Pillow", "pygame", "pymunk", "scikit-learn", "scipy", "sklearn"],
)
