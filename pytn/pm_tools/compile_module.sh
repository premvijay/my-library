#!/bin/bash
cd ~/project-alpha-peak/code/library/pm_tools/
pwd
source ~/.bashrc && conda activate conforg
# alias python3-config='python3.7m-config'
export pybind_include=./src/pybind11/include/
python3-config --includes
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3-config --includes` -I $pybind_include ./src/grid_assign_pybind11.cpp -o ./particles_to_grid`python3-config --extension-suffix`
# g++ -O3 -Wall -shared -std=c++11 -fPIC `python3-config --includes` -I $pybind_include ./src/particle_selection_pybind11.cpp -o ./select_particles`python3-config --extension-suffix`