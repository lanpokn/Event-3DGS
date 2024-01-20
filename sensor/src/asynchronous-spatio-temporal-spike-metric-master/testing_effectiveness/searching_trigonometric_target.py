"""
Function: searching a moving target in spike cube based on distance metric, which's trajectories are trigonometric function.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Mar. 5th, 2019.

"""

import numpy as np
import pickle
from event_process import event_processing
from spike_metric.cubes_3d_kernel import cubes_3d_kernel_distance
from spike_metric.spike_cube_metric import cube_using_spike_train


def initial_search_cube(initial_center, spike_cube, cube_width=10, cube_height=10, cube_length=100):
    """
   Initial search spike cube.

    Inputs:
    -------
        initial_center    - the initial center in spike cube.
        spike_cube   - the spike cube including events.

     Outputs:
    -------
        initial_cube    - the initial searching cube.
    """

    initial_search_cube = spike_cube[int(initial_center[1] - cube_width/2):int(initial_center[1] + cube_width/2),
                          int(initial_center[2] - cube_height/2):int(initial_center[2] + cube_height/2), 0:cube_length - 1]

    initial_cube = np.vstack((initial_search_cube[np.nonzero(initial_search_cube)],
                                       np.array(np.nonzero(initial_search_cube))[2],
                                       np.array(np.nonzero(initial_search_cube))[0],
                                       np.array(np.nonzero(initial_search_cube))[1]))

    return initial_cube


def search_min_center(initial_center, initial_search_events, spike_cube, cube_width=10, cube_height=10, cube_length=100, search_width=8, search_height=8):
    """
    Searching a moving target using distance metric, then return a minimum center.

    Inputs:
    -------
        initial_center    - the initial center in spike cube.
        spike_cube   - the spike cube including events.
        cube_width, cube_height, cube_length   - the width, height and length in the searching cube.
        search_width, search_height    - the width and height of the searching area in spatial domain.

    Outputs:
    -------
        minimum_center    - the searching center in spike cube, which serves as initial center in next search.
        minimum_cube    - the minimum distance cube, which serves for initial cube.

    """
    search_areas = []
    center_areas = []

    # initial_search_cube = spike_cube[int(initial_center[1] - cube_width/2):int(initial_center[1] + cube_width/2),
    #                       int(initial_center[2] - cube_height/2):int(initial_center[2] + cube_height/2), 0:cube_length - 1]
    #
    # initial_search_events = np.vstack((initial_search_cube[np.nonzero(initial_search_cube)],
    #                                    np.array(np.nonzero(initial_search_cube))[2],
    #                                    np.array(np.nonzero(initial_search_cube))[0],
    #                                    np.array(np.nonzero(initial_search_cube))[1]))

    # searching in designed areas.
    for i in range(-search_width, search_width):
        for j in range(-search_height, search_height):

            search_cube = spike_cube[int(initial_center[1] - cube_width / 2 + i):int(initial_center[1] + cube_width / 2 + i),
                          int(initial_center[2] - cube_height / 2 + j):int(initial_center[2] + cube_height / 2 + j), 0:cube_length - 1]
            search_events = np.vstack((search_cube[np.nonzero(search_cube)], np.array(np.nonzero(search_cube))[2],
                                       np.array(np.nonzero(search_cube))[0], np.array(np.nonzero(search_cube))[1]))


            search_center = np.array([initial_center[0], initial_center[1] + i, initial_center[2] + j])

            # print('search_t={}, search_x={}, search_y={}'.format(initial_center[0], initial_center[1] + i, initial_center[2] + j))

            search_areas.append(search_events)
            center_areas.append(search_center)


    # computing distances and return center of the minimum distance.
    distances = np.zeros(len(search_areas))
    for k in range(len(search_areas)):

        if len(search_areas[k][0]) == 0 or initial_search_events.shape[1] == 0:
            distances[k] = float('Inf')

        else:

            # distances[k] = cubes_3d_kernel_distance(initial_search_events, search_areas[k], cube_width, cube_height, cube_length) # spike cube using 3d kernel

            # print('initial_events_numbers={}, search_events_numbers={}'.format(len(search_areas[k][0]), initial_search_events.shape[1]))

            distances[k] = cube_using_spike_train(initial_search_events, search_areas[k], cube_width, cube_height, cube_length, cube_length) # spike train


    # select search center.
    index = np.argmin(abs(distances))

    minimum_center = center_areas[index]
    minimum_cube = search_areas[index]

    if minimum_center.shape[0]==0:
        print('The search cube has been error!')

    return minimum_center, minimum_cube


def search_trigonometric_trajectory(true_target, events, cube_width, cube_height, cube_length, search_width, search_height):
    """
    Searching a moving target using distance metric, then return trajectory error.

    Inputs:
    -------
        true_target    - the ground truth of moving trajectory.
        events    - events include polarity, timestamp, x and y.
        cube_width, cube_height, cube_length   - the width, height and length in the searching cube.
        search_width, search_height    - the width and height of the searching area in spatial domain.

    Outputs:
    -------
        search_centers    - the searching center in event stream.
        trajectory_errors    - the errors of trajectory search.

    """



    spike_cubes = event_processing.events_to_spike_cubes(events, 128, 128, 128, 128, cube_length)  # events stream to spike cubes/frames.

    # initial search cube and center.
    first_spike_cube = event_processing.events_cube_matrix(spike_cubes[0], 128, 128, cube_length)
    initial_cube = initial_search_cube(true_target[:, 0], first_spike_cube, cube_width=cube_width, cube_height=cube_height, cube_length=cube_length)
    initial_center = true_target[:, 0]

    # search moving trajectory.
    search_centers = []
    search_cubes = []
    search_centers.append(initial_center)
    search_cubes.append(initial_cube)
    first_center = np.copy(initial_center)


    for i in range(1, len(spike_cubes)):

        spike_cube = event_processing.events_cube_matrix(spike_cubes[i], 128, 128, cube_length)
        search_center, search_cube = search_min_center(initial_center, initial_cube, spike_cube, cube_width=cube_width,
                                                       cube_height=cube_height, cube_length=cube_length, search_width=search_width, search_height=search_height)

        # center operation.
        search_center[0] = search_center[0] + cube_length

        for j in range(1, 2):

            if abs(search_center[j] - first_center[j]) > cube_width + search_width:
                search_center[j] = first_center[j]

        # print('right_search_center={}'.format(search_center))

        # recursive search.
        initial_center = search_center
        # initial_cube = search_cube

        search_centers.append(search_center)
        search_cubes.append(search_cube)

    return search_centers


def computing_trajectory_error(true_target, search_centers):
    """
    Searching a moving target using distance metric, then return trajectory error.

    Inputs:
    -------
        true_target    - the ground truth of moving trajectory, using numpy matrix.
        search_centers    - the search trajectory using distance metrics.

    Outputs:
    -------
        trajectory_errors    - the errors of trajectory search.
        mean_error    - the mean error of trajectory search.

    """

    search_trajectory = np.zeros((3, len(search_centers)))

    for i in range(len(search_centers)):

        search_trajectory[:,i] = search_centers[i]

    trajectory_differences =np.vstack((true_target[0,:], true_target[1, :]- search_trajectory[1, :], true_target[2, :]- search_trajectory[2, :]))
    trajectory_errors = np.vstack((true_target[0, :], np.sqrt(np.square(trajectory_differences[1, :])+np.square(trajectory_differences[2, :]))))
    mean_error = np.mean(trajectory_errors[1,:])

    return trajectory_errors, mean_error
