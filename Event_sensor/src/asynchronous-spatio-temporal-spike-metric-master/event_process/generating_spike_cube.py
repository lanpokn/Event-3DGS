"""
Function: Random generating spike cubes including events, and an event has four elements - polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Mar. 2nd, 2018.

"""

import numpy as np
import random
from event_process import show_events
import pickle
import math
from event_process import read_dvs
from event_process import spike_cube_processing


def random_spike_cubes(spike_numbers, ratio, x_coordinate_length, y_coordinate_length, temporal_length):
    """
    random generating spike cube using AER.

    Inputs:
    -------
        spike_numbers    - the total spike numbers in spatial-temporal cube.
        ratio    - the ON/OFF polarity ratio in spike cube.
        x_coordinate_length    - x coordinate in spike cube.
        y_coordinate_length    - y coordinate in spike cube.
        temporal_length    - temporal coordinate in spike cube.

    Outputs:
    -------
        events    - the random generating events in spike cube using AER.

    """

    ON_events_numbers = int(spike_numbers * (ratio / (1 + ratio)))
    ON_events = np.ones(ON_events_numbers)
    OFF_events = - np.ones(spike_numbers - ON_events_numbers)
    polarity_events = np.hstack((ON_events, OFF_events))

    # no replacement.
    # x_events = random.sample(range(0, x_coordinate_length), spike_numbers)
    # y_events = random.sample(range(0, y_coordinate_length), spike_numbers)
    # t_events = random.sample(range(0, temporal_length), spike_numbers)

    # available replacement.
    x_events = np.array([random.randint(0, x_coordinate_length-1) for _ in range(spike_numbers)])
    y_events = np.array([random.randint(0, y_coordinate_length-1) for _ in range(spike_numbers)])
    t_events = np.array([random.randint(0, temporal_length-1) for _ in range(spike_numbers)])


    events = np.vstack((polarity_events, t_events, x_events, y_events))
    events = events[:, events[1, :].argsort()]  # sorted by t coordinate

    return events


def generating_2d_trigonometric_events(time_length=10000, spike_numbers=500, center=64, offset=3, ratio=1):
    """
    generating an moving target using 2D trigonometric function.

    Inputs:
    -------
        time_length    - the time length.
        spike_numbers    - the total spike numbers in spatial-temporal cube.
        center    - the center ofx or y coordinate.
        offset    - the x or y coordinate offset around center.
        ratio    - the ON/OFF polarity ratio in spike cube.

    Outputs:
    -------
        events    - the random generating events in spike cube using AER.

    """

    ON_events_numbers = int(spike_numbers * (ratio / (1 + ratio)))
    ON_events = np.ones(ON_events_numbers)
    OFF_events = - np.ones(spike_numbers - ON_events_numbers)

    polarity_events = np.hstack((ON_events, OFF_events))
    t_events = np.array([random.randint(0, time_length - 1) for _ in range(spike_numbers)])
    x_events = np.rint(center + 15 * np.sin((t_events / 1500) * math.pi) + random.sample(range(-offset, offset), 1)[0])
    y_events = np.array([random.randint(center - offset, center + offset) for _ in range(spike_numbers)])

    events = np.vstack((polarity_events, t_events, x_events, y_events))
    events = events[:, events[1, :].argsort()]  # sorted by t coordinate.

    t_events = np.linspace(0, time_length, 51)[0:50]
    true_target = np.vstack((t_events, np.rint(center + 15 * np.sin((t_events / 1500) * math.pi)), center * np.ones(len(t_events))))

    #true_target = np.vstack((t_events, np.rint(center + 5 * np.sin(t_events / 500) * (math.pi)), center*np.ones(spike_numbers)))
    true_target = true_target[:, true_target[0, :].argsort()]  # sorted by t coordinate.

    return events, true_target


def generating_3d_trigonometric_events(time_length=10000, spike_numbers=500, center=64, offset=3, ratio=1):
    """
   generating an moving target using 2D trigonometric function.

   Inputs:
   -------
       time_length    - the time length.
       spike_numbers    - the total spike numbers in spatial-temporal cube.
       center    - the center ofx or y coordinate.
       offset    - the x or y coordinate offset around center.
       ratio    - the ON/OFF polarity ratio in spike cube.

   Outputs:
   -------
       events    - the random generating events in spike cube using AER.

   """

    ON_events_numbers = int(spike_numbers * (ratio / (1 + ratio)))
    ON_events = np.ones(ON_events_numbers)
    OFF_events = - np.ones(spike_numbers - ON_events_numbers)

    polarity_events = np.hstack((ON_events, OFF_events))
    t_events = np.array([random.randint(0, time_length - 1) for _ in range(spike_numbers)])
    # x_events = np.rint(center + 8 * np.sin(t_events / 250) * (math.pi) + random.sample(range(-offset, offset), 1)[0])
    # y_events = np.rint(center + 8 * np.cos(t_events / 250) * (math.pi) + random.sample(range(-offset, offset), 1)[0])

    x_events = np.rint(center + 22 * np.sin((t_events / 800) * math.pi) + random.sample(range(-offset, offset), 1)[0]) # 15
    y_events = np.rint(center + 22 * np.cos((t_events / 800) * math.pi) + random.sample(range(-offset, offset), 1)[0]) # 15


    events = np.vstack((polarity_events, t_events, x_events, y_events))
    events = events[:, events[1, :].argsort()]  #

    t_events = np.linspace(0, time_length, 51)[0:50]
    true_target = np.vstack((t_events, np.rint(center + 22 * np.sin((t_events / 800) * math.pi)), np.rint(center + 22 * np.cos((t_events / 800) * math.pi))))

    #true_target = np.vstack((t_events, np.rint(center + 8 * np.sin(t_events / 250) * (math.pi)), np.rint(center + 8 * np.cos(t_events / 250) * (math.pi))))
    true_target = true_target[:, true_target[0, :].argsort()]  # sorted by t coordinate.

    return events, true_target



if __name__ == '__main__':


    ### random generating events in spike cube.
    # spike_numbers = 100
    # time_length = 1000
    # ratio = 0

    # events = random_spike_cubes(spike_numbers, ratio, 128, 128, time_length)
    # show_events.show_ON_OFF_events(events, width=128, height=128, length=max(events[1, :]))
    # save the pkl format.
    # pickle.dump(events, open('../datasets/simulating_dataset/events_{}_{}_{}.pkl'.format(spike_numbers, ratio, time_length), 'wb'))
    # pkl_events = open('../datasets/simulating_dataset/events_{}_{}.pkl'.format(spike_numbers, time_length),'rb')
    # read_events = pickle.load(pkl_events)


    ### generating an moving target using 2D or 3D trigonometric function.
    events, true_target = generating_2d_trigonometric_events(time_length=10000, spike_numbers=500, center=64, offset=3, ratio=1)
    # events, true_target = generating_3d_trigonometric_events(time_length=10000, spike_numbers=1000, center=64, offset=8, ratio=1)
    show_events.show_simulating_events(events, width=128, height=128, length=max(events[1, :]))
    new_events = spike_cube_processing.events_increasing_noises(events, 0, 0.1, 128, 128) #0.1
    show_events.show_simulating_events(new_events, width=128, height=128, length=max(events[1, :]))

    # save the pkl format.
    pickle.dump(events, open('../datasets/simulating_dataset/events_3d_trigonometric.pkl', 'wb'))
    pickle.dump(true_target, open('../datasets/simulating_dataset/events_3d_true_target.pkl', 'wb'))