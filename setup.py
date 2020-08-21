import setuptools
from distutils.core import setup
# from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dino",
    version="0.0.1",
    author="Alexandre Manoury",
    author_email="alex@pika.tf",
    description="Active Learning Environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pikalchemist/dino",
    # ext_modules=cythonize("lems/lib/operations.pyx"),
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)