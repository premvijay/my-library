import numpy as np
from time import time
import sys


def average_2D_slice(grid, box_size, axis, around_position, thick):
    t_now = time()
    # print(box_size, axis, around_position, thick)
    if around_position == 'centre':
        start = round((1/2 - thick/2 / box_size) * grid.shape[axis])
        stop = round((1/2 + thick/2 / box_size) * grid.shape[axis])
        idx_range = slice(start,stop) 
    else:
        start = (around_position[axis] - thick/2) * grid.shape[axis] // box_size
        stop = (around_position[axis] + thick/2) * grid.shape[axis] // box_size
        # print(start,stop, flush=True)
        idx_range = np.arange(start,stop,dtype='int') % grid.shape[axis]

    idx_range_3D = tuple([slice(None)]* axis + [idx_range])
    # print(idx_range_3D, flush=True)
    t_bef, t_now = t_now, time()
    print("\n    slice index obtained")
    print(t_now-t_bef)
    # print(grid[idx_range_3D])
    # sys.stdout.flush()
    return grid[idx_range_3D].mean(axis=axis)

    # grid_select = grid[idx_range_3D]
    # t_bef, t_now = t_now, time()
    # print("    thich slice grid selected")
    # print(t_now-t_bef)
    # grid_select = grid[:,:,idx_range]
    # t_bef, t_now = t_now, time()
    # print("    thich slice grid selected simply")
    # print(t_now-t_bef)
    # grid2D = grid_select.mean(axis=axis)
    # t_bef, t_now = t_now, time()
    # print("    thick slice pressed")
    # print(t_now-t_bef)
    # sys.stdout.flush()
    # return grid2D
