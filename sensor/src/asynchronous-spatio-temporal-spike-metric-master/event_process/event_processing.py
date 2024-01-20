"""
Function: Events processing basic library for dynamic vision sensor(DVS).
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, May 10th, 2018
"""

import numpy as np
import math

def aer_events(aer_data):
    """
    aer_data to events in dynamic vision sensor.

    Inputs:
    -------
        aer_data    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).

    Outputs:
    -------
        events    - the Matrix dataset.

    """
    events = np.zeros((4, len(aer_data.ts)))
    events[0, :] = aer_data.t *2 -1
    events[1, :] = aer_data.ts
    events[2, :] = aer_data.x
    events[3, :] = aer_data.y

    return events


def events_to_cubes(events, time_interval):
    """
    events are split into cubes.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        time_interval    - the split time interval.
   Outputs:
    -------
        events_cubes    - the cubes of events.

    """

    events_cubes = [[] for _ in range(math.ceil(max(events[1, :] / time_interval)))]

    for i in range(events.shape[1]):
        k = math.floor(events[1, i] / time_interval)
        events_cubes[k].append(events[:, i])

    return events_cubes


def events_to_spike_cubes(events, width, height, x_cube_size, y_cube_size, t_cube_size):
    """
    events are split into spike cubes.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        width, height    - the width and height resolutions of dynamic vision sensor.
        x_cube_size, y_cube_size, t_cube_size    - the width, height and temporal size of spike cubes.
   Outputs:
    -------
        events_cubes    - the cubes of events.

    """

    num = int((width/x_cube_size)*(height/y_cube_size)*(math.ceil(max(events[1, :] / t_cube_size))))
    events_cube = [[] for _ in range(num)]
    #print('num={}'.format(num))

    for i in range(events.shape[1]):

        k = math.floor(events[2, i]/x_cube_size) + math.floor(events[3, i]/y_cube_size)*int(width/x_cube_size) + math.floor(events[1, i]/t_cube_size)*int(width/x_cube_size)*int(height/y_cube_size)

        #print('t_cube_size={}, k={}, i={}, event={}, feature={}'.format(t_cube_size, k, i, events[:, i], math.floor(events[1, i]/t_cube_size)))
        events_cube[k].append(events[:, i])


    return events_cube


def events_cube_matrix(events_cube, width, height, time_interval):
    """
    events are split into cubes.

    Inputs:
    -------
        events_cube    - the events in time interval.
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        time_interval   - the time interval.
   Outputs:
    -------
        spike_cube    - the spike cube of events.

    """
    spike_cube = np.zeros((width, height, time_interval))

    for i in range(len(events_cube)):

        # if int(events_cube[i][2])==0 or int(events_cube[i][3]) == 0:
        #     print('x={}'.format(events_cube[i][2]))
        #     print('y={}'.format(events_cube[i][3]))

        # print('x={},y={},t={},p={}'.format(int(events_cube[i][2]%width-1), int(events_cube[i][3]%height-1), int(events_cube[i][1] - events_cube[0][1]), events_cube[i][0]))

        spike_cube[int(events_cube[i][2]%width-1), int(events_cube[i][3]%height-1), int(events_cube[i][1] - events_cube[0][1])] = events_cube[i][0]


    return spike_cube


def events_to_cube(events_cube, width, height, time_interval):
    """
    events are split into cubes.

    Inputs:
    -------
        events_cube    - the events in time interval.
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        time_interval   - the time interval.
   Outputs:
    -------
        spike_cube    - the spike cube of events.

    """
    spike_cube = np.zeros((width, height, time_interval))

    for i in range(events_cube.shape[1]):

        # if int(events_cube[2][i])==0 or int(events_cube[3][i]) == 0:
        #     print('x={}'.format(events_cube[2][i]))
        #     print('y={}'.format(events_cube[3][i]))

        spike_cube[int(events_cube[2][i]%width-1), int(events_cube[3][i]%height-1), int(events_cube[1][i] - events_cube[1][0])] = events_cube[0][i]


    return spike_cube


def aer_on_off_events(aer_data):
    """
    separation ON & OFF events in dynamic vision sensor.

    Inputs:
    -------
        aer_data    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).

    Outputs:
    -------
        events_ON    - ON events in the increasing intensity.
        events_OFF    - OFF events in the decreasing intensity.

    """
    events_ON = np.zeros((4, np.count_nonzero(aer_data.t == 1)))
    events_OFF = np.zeros((4, np.count_nonzero(aer_data.t == 0)))
    index_ON = np.where(aer_data.t == 1)[0]
    index_OFF = np.where(aer_data.t == 0)[0]

    # save ON events for AER sensor
    events_ON[0, :] = aer_data.t[index_ON]
    events_ON[1, :] = aer_data.ts[index_ON]
    events_ON[2, :] = aer_data.x[index_ON]
    events_ON[3, :] = aer_data.y[index_ON]

    # save ON events for AER sensor
    events_OFF[0, :] = aer_data.t[index_OFF]
    events_OFF[1, :] = aer_data.ts[index_OFF]
    events_OFF[2, :] = aer_data.x[index_OFF]
    events_OFF[3, :] = aer_data.y[index_OFF]

    return events_ON, events_OFF


def On_off_events(events):
    """
    separation ON & OFF events in dynamic vision sensor.

    Inputs:
    -------
        events    - the matrix dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).

    Outputs:
    -------
        events_ON    - ON events in the increasing intensity.
        events_OFF    - OFF events in the decreasing intensity.

    """
    events_ON = np.zeros((4, np.count_nonzero(events[0, :] == 1)))
    events_OFF = np.zeros((4, np.count_nonzero(events[0, :] == -1)))
    index_ON = np.where(events[0, :] == 1)[0]
    index_OFF = np.where(events[0, :] == -1)[0]

    # save ON events for AER sensor
    events_ON[0, :] = events[0, :][index_ON]
    events_ON[1, :] = events[1, :][index_ON]
    events_ON[2, :] = events[2, :][index_ON]
    events_ON[3, :] = events[3, :][index_ON]

    # save ON events for AER sensor
    events_OFF[0, :] = events[0, :][index_OFF]
    events_OFF[1, :] = events[1, :][index_OFF]
    events_OFF[2, :] = events[2, :][index_OFF]
    events_OFF[3, :] = events[3, :][index_OFF]

    return events_ON, events_OFF


def select_aer_events(events, time_length):
    """
        Selecting time length for aer_data to events in dynamic vision sensor.

        Inputs:
        -------
            events    - the matrix dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
            time_length    - the selecting time length.

        Outputs:
        -------
            select_aer_data    - the selecting aer data.

    """
    index = np.searchsorted(events[1, :], time_length)

    select_aer_data = events[:, 0:index]

    return select_aer_data


def separating_on_off_events(aer_data):
    """
    separation ON & OFF events in dynamic vision sensor.

    Inputs:
    -------
        aer_data    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).

    Outputs:
    -------
        events_ON    - ON events in the increasing intensity.
        events_OFF    - OFF events in the decreasing intensity.

    """
    events_ON = np.zeros((4, np.count_nonzero(aer_data.t == 1)))
    events_OFF = np.zeros((4, np.count_nonzero(aer_data.t == 0)))
    index_ON = np.where(aer_data.t == 1)[0]
    index_OFF = np.where(aer_data.t == 0)[0]

    # save ON events for AER sensor
    events_ON[0, :] = aer_data.t[index_ON]
    events_ON[1, :] = aer_data.ts[index_ON]
    events_ON[2, :] = aer_data.x[index_ON]
    events_ON[3, :] = aer_data.y[index_ON]

    # save ON events for AER sensor
    events_OFF[0, :] = aer_data.t[index_OFF]
    events_OFF[1, :] = aer_data.ts[index_OFF]
    events_OFF[2, :] = aer_data.x[index_OFF]
    events_OFF[3, :] = aer_data.y[index_OFF]

    return events_ON, events_OFF

def separating_ON_OFF_events(events):
    """
    separation ON & OFF events in dynamic vision sensor.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).

    Outputs:
    -------
        events_ON    - ON events in the increasing intensity.
        events_OFF    - OFF events in the decreasing intensity.

    """
    index_ON = np.where(events[3,:]==1)
    index_OFF = np.where(events[3,:]==-1)
    ON_events = events[:, index_ON]
    OFF_events = events[:, index_OFF]

    return ON_events, OFF_events