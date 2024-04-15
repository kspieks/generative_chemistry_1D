import os

from setuptools import find_packages, setup

__version__ = None

# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="gen_chem_1D",
    version=__version__,
    author="Kevin Spiekermann",
    description="This codebase uses generative models to create SMILES strings and uses predictive models to bias the generators towards better molecular properties.",
    url="https://github.com/kspieks/generative_chemistry_1D",
    packages=find_packages(),
    long_description=read('README.md'),
)
