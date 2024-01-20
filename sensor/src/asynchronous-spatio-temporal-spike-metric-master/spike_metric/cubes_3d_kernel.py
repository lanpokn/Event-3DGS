"""
Function: 3d Gaussian kernel method for spike cubes.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Feb. 16th, 2019.

"""

import numpy as np

def cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma):
    """
    Computing inner product between spike cubes using 3d gaussian kernel method.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.

    Outputs:
    -------
        inner_product    - the inner product between events and new_events.

    """
    #print('events number={}'.format(len(events[0,:])))
    #print('ON number={}'.format(np.sum(events[0, :]==1)))
    ON_scale = np.sum(events[0, :]==1)/(len(events[0, :])) # ON events in history
    # new_OFF_scale = np.sum(new_events[0, :]==-1)/len(events[0, :]) # ON new events in history
    new_ON_scale = np.sum(new_events[0, :] == 1) / (len(new_events[0, :]))  # ON new events in history

    # print('events_numbers={}, new_events_numbers={}'.format(len(events[0, :]), len(new_events[0, :])))

    polarity_scale = ON_scale*new_ON_scale + (1-ON_scale)*(1-new_ON_scale)
    # polarity_scale = 1 + abs(ON_scale-new_OFF_scale) # simply polarity for integrated formulation.

    x_index = events[2, :][:, None] - new_events[2, :][None, :]
    y_index = events[3, :][:, None] - new_events[3, :][None, :]
    t_index = events[1, :][:, None] - new_events[1, :][None, :]

    dist_matrix = np.exp(- x_index**2 / (2*x_sigma**2) - y_index**2 / (2*y_sigma**2) - t_index**2 / (2*t_sigma**2))

    inner_product = polarity_scale * np.sum(dist_matrix)

    return inner_product


def cubes_3d_kernel_distance(events, new_events, x_sigma, y_sigma, t_sigma):
    """
    Computing distance between spike cubes using inner product in RKHS.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.

    Outputs:
    -------
        distance    - the distance between events and new_events.

    """

    if len(np.transpose(events)) <= 5 or len(np.transpose(events)) <= 5:
        distance = 0
    else:

        distance = cubes_3d_kernel_method(events, events, x_sigma, y_sigma, t_sigma)
        distance += cubes_3d_kernel_method(new_events, new_events, x_sigma, y_sigma, t_sigma)
        distance -= 2 * cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma)

    return distance

