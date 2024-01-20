"""
Function: spike cube measurement function.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Jan. 24th, 2018.

"""
import numpy as np
from event_process import event_processing
from spike_metric import spike_train_metric, fft_3d_convolution
from spike_metric import cubes_3d_kernel


def kernel_method_spike_train(events, new_events, x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000):
    """
        kernel method  for spike train, such as polarity independent and polarity interference. (Hamming distance)

        Inputs:
        -------
            events    - events include polarity, timestamp, x and y.
            new_events    - events after changing operation.
            x_cube_size, y_cube_size, t_cube_size   - the spike cube parameters.
            sigma  - the parameter of gaussian kernel.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """

    events_cubes = event_processing.events_to_cubes(events, t_cube_size)
    new_events_cubes = event_processing.events_to_cubes(new_events, t_cube_size)

    distance = 0
    for k in range(len(events_cubes)):

        spike_cube = event_processing.events_cube_matrix(events_cubes[k], x_cube_size, y_cube_size, t_cube_size)
        new_spike_cube = event_processing.events_cube_matrix(new_events_cubes[k], x_cube_size, y_cube_size, t_cube_size)

        for i in range(x_cube_size):
            for j in range(y_cube_size):

                if len(np.nonzero(spike_cube[i,j,:])[0])==0 and len(np.nonzero(new_spike_cube[i, j, :])[0])==0:
                    distance += 0

                else:
                    distance += spike_train_metric.cal_dist_pid(spike_cube[i,j,:], new_spike_cube[i,j,:], sigma) # polarity independent measurement
                    # distance += spike_train_metric.cal_dist_pif(spike_cube[i,j,:], new_spike_cube[i,j,:], sigma) # polarity interference measurement
                    # distance += spike_train_metric.hamming_distance(spike_cube[i,j,:], new_spike_cube[i,j,:]) # Hamming distance

    # return distance/(x_cube_size*y_cube_size)
    return distance


def cube_using_spike_train(events_cube, new_events_cube, x_cube_size, y_cube_size, t_cube_size, sigma):
    """
        kernel method  for spike train.

        Inputs:
        -------
            events_cube    - events include polarity, timestamp, x and y.
            new_events_cube    - events after changing operation.
            x_cube_size, y_cube_size, t_cube_size   - the spike cube parameters.
            sigma  - the parameter of gaussian kernel.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """

    distance = 0

    spike_cube = event_processing.events_to_cube(events_cube, x_cube_size, y_cube_size, t_cube_size)
    new_spike_cube = event_processing.events_to_cube(new_events_cube, x_cube_size, y_cube_size, t_cube_size)

    for i in range(x_cube_size):
        for j in range(y_cube_size):

            if len(np.nonzero(spike_cube[i, j, :])[0]) == 0 and len(np.nonzero(new_spike_cube[i, j, :])[0]) == 0:
                distance += 0

            else:
                distance += spike_train_metric.cal_dist_pid(spike_cube[i, j, :], new_spike_cube[i, j, :], sigma)  # polarity independent measurement
                # distance += spike_train_metric.cal_dist_pif(spike_cube[i,j,:], new_spike_cube[i,j,:], sigma) # polarity interference measurement
                # distance += spike_train_metric.hamming_distance(spike_cube[i,j,:], new_spike_cube[i,j,:]) # Hamming distance

    return distance


def fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32, t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100):
    """
        Computing distance between spike cubes using 3d convolution in l1 norm, which can be decomposed on fft(fast fourier transform).

        Inputs:
        -------
            events    - events include polarity, timestamp, x and y.
            new_events    - events after changing operation.
            x_cube_size, y_cube_size, t_cube_size   - the spike cube parameters.
            x_sigma, y_sigma, z_sigma  - the parameters of 3d gaussian kernel.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """

    gaussian_3d = fft_3d_convolution.get_3d_gaussian_kernel(x_size=20, y_size=20, t_size=500, x_sigma=x_sigma, y_sigma=y_sigma, t_sigma=t_sigma) # 3d gaussian function

    # events_cubes = event_processing.events_to_cubes(events, t_cube_size)
    # new_events_cubes = event_processing.events_to_cubes(new_events, t_cube_size)

    events_cubes = event_processing.events_to_spike_cubes(events, 128, 128, x_cube_size, y_cube_size, t_cube_size)
    new_events_cubes = event_processing.events_to_spike_cubes(new_events, 128, 128, x_cube_size, y_cube_size,
                                                              t_cube_size)

    distance = 0
    for k in range(len(events_cubes)):
    # for k in range(2):

        spike_cube = event_processing.events_cube_matrix(events_cubes[k], x_cube_size, y_cube_size, t_cube_size)
        new_spike_cube = event_processing.events_cube_matrix(new_events_cubes[k], x_cube_size, y_cube_size, t_cube_size)

        inverse_fft = fft_3d_convolution.fft_convolution(spike_cube, gaussian_3d)
        new_inverse_fft = fft_3d_convolution.fft_convolution(new_spike_cube, gaussian_3d)

        distance_matrix = abs(inverse_fft - new_inverse_fft) #l1 norm

        distance += distance_matrix.sum()

    return distance


def kernel_method_spike_cubes(events, new_events, width=128, height=128, x_cube_size=32, y_cube_size=32, t_cube_size=5000, x_sigma=5, y_sigma=5, t_sigma=5000):
    """
        3d gaussian kernel method  for spike cubes, such as polarity independent and polarity interference.

        Inputs:
        -------
            events    - events include polarity, timestamp, x and y.
            new_events    - events after changing operation.
            width, height  - the width and height of dynamic vision sensor.
            x_cube_size, y_cube_size, t_cube_size  - the size of spike cube.
            x_sigma, y_sigma, t_sigma  - the 3d gaussian kernel parameters.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """

    events_cube = event_processing.events_to_spike_cubes(events, width, height, x_cube_size, y_cube_size, t_cube_size)
    new_events_cubes = event_processing.events_to_spike_cubes(new_events, width, height, x_cube_size, y_cube_size, t_cube_size)

    distance = 0
    for k in range(0, min(len(events_cube), len(new_events_cubes))):

        events_data = np.transpose(np.array(events_cube[k]))
        new_events_data = np.transpose(np.array(new_events_cubes[k]))

        if len(events_data)==0 and len(new_events_data)==0:
            distance += 0

        else:
            distance += cubes_3d_kernel.cubes_3d_kernel_distance(events_data, new_events_data, x_sigma, y_sigma, t_sigma)

    return distance