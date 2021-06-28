from glob import glob
from distutils.core import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "particles_to_grid",
        ["src/grid_assign_pybind11.cpp"]),
    ),
    Pybind11Extension(
        "select_particles",
        ["src/particle_selection_pybind11.cpp"]),
    ),
]

setup(
    name="pm_tools",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)