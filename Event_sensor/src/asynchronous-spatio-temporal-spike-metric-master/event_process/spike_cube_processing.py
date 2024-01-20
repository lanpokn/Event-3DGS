"""
Function: spike cube processing --- cyclic displacement, removing and spatial-temporal random changing.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Jan. 23th, 2019.

"""
import numpy as np
import random
from event_process.generating_spike_cube import random_spike_cubes
import math

def center_rotation(events, rotation_theta, width, height):
    """
   Spike cube rotation.

   Inputs:
   -------
   events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
   rotation_theta  - spike cube around center coordination rotation.
   width, height    - the array size of event camera.

   Outputs:
   -------
       new_events    - the new events by rotation operation.
   """
    rotation_theta = rotation_theta / 180 * (math.pi)
    new_events = np.copy(events)
    center_cordinate = [width / 2-10, height / 2-10] # w/2, h/2-10
    for i in range(len(events[1])):
        # x_new = x*cos(theta) - y*sin(theta)
        if (events[2][i] - center_cordinate[0]) * math.cos(rotation_theta) - (
                events[3][i] - center_cordinate[1]) * math.sin(rotation_theta) + center_cordinate[0] > width:
            new_events[2][i] = width - 1
        if (events[2][i] - center_cordinate[0]) * math.cos(rotation_theta) - (
                events[3][i] - center_cordinate[1]) * math.sin(rotation_theta) + center_cordinate[0] < 0:
            new_events[2][i] = 0
        else:
            new_events[2][i] = (events[2][i] - center_cordinate[0]) * math.cos(rotation_theta) - (
                        events[3][i] - center_cordinate[1]) * math.sin(rotation_theta) + center_cordinate[0]

        # y_new = y*cos(theta) + x*sin(theta)
        if (events[3][i] - center_cordinate[1]) * math.cos(rotation_theta) + (
                events[2][i] - center_cordinate[0]) * math.sin(rotation_theta) + center_cordinate[1] > height:
            new_events[3][i] = height - 1
        if (events[3][i] - center_cordinate[1]) * math.cos(rotation_theta) + (
                events[2][i] - center_cordinate[0]) * math.sin(rotation_theta) + center_cordinate[1] < 0:
            new_events[3][i] = 0
        else:
            new_events[3][i] = (events[3][i] - center_cordinate[1]) * math.cos(rotation_theta) + (
                        events[2][i] - center_cordinate[0]) * math.sin(rotation_theta) + center_cordinate[1]

    return new_events


def spatial_temporal_trans(events, max_width, max_height):
    """
    image plane translation.

    Inputs:
    -------
    events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
    max_width    - the maximum spatial changing size.
    max_height   - the maximum temporal changing size.

    Outputs:
    -------
        new_events    - the new events by spatial-temporal random change.
    """

    new_events = np.copy(events)
    max_timestamp = max(events[1])

    for i in range(len(events[1])):
        if new_events[2][i] - int(max_width / max_timestamp * events[1][i]) <= 0:
            new_events[2][i] = 0
        else:
            new_events[2][i] = new_events[2][i] - int(max_width / max_timestamp * events[1][i])

        if new_events[3][i] - int(max_height / max_timestamp * events[1][i]) <= 0:
            new_events[3][i] = 0
        else:
            new_events[3][i] = new_events[3][i] - int(max_height / max_timestamp * events[1][i])

    return new_events

def temporal_cyclic_displacement(events, k):
    """
    Temporal cyclic displacement for spike cubes.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        k    - the cyclic displacement step k.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    index = np.searchsorted(events[1], max(events[1])-k)
    new_events = np.hstack((events[:, index:], events[:, :index]))
    new_events[1] = np.hstack((events[1, index:]-int(max(events[1]))+k, events[1, :index]+k))

    return new_events


def spatial_cyclic_displacement(events, width, k):
    """
    Spatial cyclic displacement for spike cubes.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        width    - the length of AER width.
        k    - the cyclic displacement step k.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    index = width-1-k
    # new_events = np.hstack((events[:, index:], events[:, :index]))
    new_events = np.copy(events)

    for i in range(len(events[2, :])): # x coordinate
        if events[2, i] > index:
            new_events[2, i] = events[2, i] + k + 1 - width
        else:
            new_events[2, i] = events[2, i] + k


    return new_events


def temporal_spatial_cyclic_displacement(events, width, temporal_step, spatial_step):
    """
    Spatial cyclic displacement for spike cubes.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        width    - the length of AER width, such as DVS128.
        temporal_step    - the temporal cyclic displacement step.
        spatial_step    - the spatial cyclic displacement step.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    temporal_index = np.searchsorted(events[1], max(events[1]) - temporal_step) + 1
    new_events = np.hstack((events[:, temporal_index:], events[:, :temporal_index]))
    new_events[1] = np.hstack((events[1, temporal_index:] - int(max(events[1])) + temporal_step, events[1, :temporal_index] + temporal_step))

    spatial_index = width - spatial_step
    spatial_temporal_events = np.copy(new_events)

    for i in range(len(new_events[2, :])):  # x coordinate
        if new_events[2, i] > spatial_index:
            spatial_temporal_events[2, i] = new_events[2, i] + spatial_step - width
        else:
            spatial_temporal_events[2, i] = new_events[2, i] + spatial_step-1

    return spatial_temporal_events


def random_remove_spikes(events, spike_numbers):
    """
    Random removing spike numbers for spike cube.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        spike_numbers    - the removing spike numbers.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    remove_indexs = random.sample(range(0, events.shape[1]), spike_numbers)

    new_events = np.delete(events, remove_indexs, axis=1)

    return new_events


def spatial_temporal_random_change(events, temporal_size, spatial_size, width, height):
    """
    spatial and temporal random change for spike cube.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        spatial_size    - the spatial random changing size, it should be ranged from 0 to 3.
        temporal_size   - the temporal random changing size.

    Outputs:
    -------
        new_events    - the new events by spatial-temporal random change.

    """

    new_events = np.copy(events)

    if temporal_size >0:
        temporal_random = np.random.randint(-temporal_size, temporal_size, size=len(events[1]))
        new_timestamp = events[1] + temporal_random
        new_events[1] = abs(new_timestamp)

    if spatial_size >0:
        spatial_random = np.random.randint(-spatial_size, spatial_size, size=len(events[2]))
        new_x_coordinate = events[2] + spatial_random
        new_y_coordinate = events[3] + spatial_random
        new_events[2] = abs(new_x_coordinate)
        new_events[3] = abs(new_y_coordinate)

    for i in range(len(events[2])):
        if new_events[2][i] >= width:
            new_events[2][i] = width-1
        if new_events[3][i] >= height:
            new_events[3][i] = height-1


    new_events = new_events[:, new_events[1, :].argsort()] # sorted by timestamp

    return new_events


def events_polarity_random_changing(events, changing_percentage):
    """
    Random changing maximum number polarity percentage in spike cube.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        changing_percentage    - the changing percentage in maximum polarity number.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    new_events = np.copy(events)

    index = np.argwhere(events[0, :] == 1) if np.sum(events[0, :] == 1) > np.sum(events[0, :] == -1) else np.argwhere(events[0, :] == -1)
    polarity_index = 1 if np.sum(events[0, :] == 1) > np.sum(events[0, :] == -1) else -1

    random_changing_numbers = int(np.sum(events[0, :]==polarity_index)*changing_percentage)
    change_index = index[random.sample(range(0, np.sum(events[0, :] == polarity_index)), random_changing_numbers)]

    new_events[0, change_index] = - polarity_index

    return new_events


def events_increasing_noises(events, SNR, ratio, width, height):
    """
    Random increasing noise in spike cube.

    Inputs:
    -------
        events    - numpy.array: the spike cube includes polarity, timestamp, x coordinate and y coordinate.
        SNR    - the ratio of signal to noise, namely spike numbers : noise numbers.
        ratio   - the ratio of polarity, namely ON:OFF.
        width, height  - the width and height of AER sensor.

    Outputs:
    -------
        new_events    - the new events by random removing events.

    """

    noise_numbers = SNR * events.shape[1]
    noise_events = random_spike_cubes(int(noise_numbers), ratio, width, height, int(max(events[1, :])))

    new_events = np.hstack((events, noise_events))
    new_events = new_events[:, new_events[1, :].argsort()]  # sorted by t coordinate

    return new_events