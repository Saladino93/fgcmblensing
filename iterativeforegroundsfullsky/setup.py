import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='itfgs',
    version='0.0.1',
    packages=['itfgs'],
    install_requires=['numpy', 'pyfftw'], #removed mpi4py for travis tests
    requires=['numpy', 'lenscarf', 'plancklens'],
    long_description=long_description)