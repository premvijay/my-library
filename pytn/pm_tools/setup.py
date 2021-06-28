from glob import glob
from distutils.core import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pm_tools",
        sorted(glob("src/*.cpp")),
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)