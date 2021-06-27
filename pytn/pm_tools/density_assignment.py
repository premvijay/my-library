# To use this run compile script

from . import particles_to_grid
import numpy as np

def assign_density(pos, box_size, grid_size = 512, scheme='CIC', shift=0, overdensity=True):
    # pos *= grid_size / box_size    # This happens without using extra memory
    # pos += shift
    assert (pos <= box_size).all()
    pos_scaled_to_grid_units = pos * grid_size / box_size + shift
    # pos_scaled_to_grid_units += shift
    # print(pos_scaled_to_grid_units)

    if scheme =='NGP':
        particle_grid = particles_to_grid.assign_ngp(pos_scaled_to_grid_units, grid_size)
    elif scheme=='CIC':
        particle_grid = particles_to_grid.assign_cic(pos_scaled_to_grid_units, grid_size)
    elif scheme=='TSC':
        particle_grid = particles_to_grid.assign_tsc(pos_scaled_to_grid_units, grid_size)
    else:
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    
    # print(density_grid)
    if overdensity:
        density_grid = particle_grid
        overdensity_grid = density_grid / (np.mean(density_grid)+1e-5) - 1
        return overdensity_grid
    else:
        return particle_grid


def assign_density_sph(pos, mass, hsml, box_size, grid_size = 512, scheme='cubic', shift=0, return_grid='density'):
    # pos *= grid_size / box_size    # This happens without using extra memory
    # pos += shift
    pos_scaled_to_grid_units = pos * grid_size / box_size + shift
    hsml_scaled_to_grid_units = hsml * grid_size / box_size
    # pos_scaled_to_grid_units += shift

    if scheme =='cubic':
        mass_grid = particles_to_grid.assign_mass_sph(pos_scaled_to_grid_units, mass, hsml_scaled_to_grid_units, grid_size)
    elif scheme=='quartic':
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    else:
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    
    density_grid = mass_grid * (grid_size / box_size)**3
    # print(density_grid)
    if return_grid=='delta':
        overdensity_grid = mass_grid / (np.mean(mass_grid)+1e-5) - 1
        return overdensity_grid
    elif return_grid=='density':
        return density_grid
    elif return_grid=='mass':
        return mass_grid