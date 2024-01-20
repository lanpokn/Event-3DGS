"""
Function: Measure distances of spike trains.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Oct. 28th, 2018.

"""
import numpy as np
from event_process import spike_train_processing
from spike_metric import spike_train_metric


def spike_train_displacement_distances(spike_train, displacement_steps, sigma):
    """
    spike train distances in cyclic displacement steps.

    Inputs:
    -------
        spike_train    - numpy.array: the spike train includes spiking firing timestamp.
        displacement_steps    - the cyclic displacement steps.
        sigma    - the kernel measure method parameter.

    Outputs:
    -------
        pif_distance    - polarity interference measurement distance.
        pid_distance    - polarity independent measurement distance.
        hamming_distance    - hamming distance.

    """
    pif_distance = np.zeros((len(displacement_steps)))
    pid_distance = np.zeros((len(displacement_steps)))
    hamming_distance = np.zeros((len(displacement_steps)))
    for i, displacement_step in enumerate(displacement_steps):
        new_spike_train = spike_train_processing.displacement_spikes(spike_train, displacement_step)
        pif_distance[i] = spike_train_metric.cal_dist_pif(spike_train, new_spike_train, sigma)  # polarity interference measurement
        pid_distance[i] = spike_train_metric.cal_dist_pid(spike_train, new_spike_train, sigma)  # polarity independent measurement
        hamming_distance[i] = spike_train_metric.hamming_distance(spike_train, new_spike_train)  # Hamming distance measurement

    return pif_distance, pid_distance, hamming_distance


def spike_train_removing_distance(spike_train, removing_steps, sigma):
    """
    spike train distances in removing steps.

    Inputs:
    -------
        spike_train    - numpy.array: the spike train includes spiking firing timestamp.
        removing_steps    - the removing steps.
        sigma    - the kernel measure method parameter.

    Outputs:
    -------
        pif_distance    - polarity interference measurement distance.
        pid_distance    - polarity independent measurement distance.
        hamming_distance    - hamming distance.

    """
    pif_distance = np.zeros((len(removing_steps)))
    pid_distance = np.zeros((len(removing_steps)))
    hamming_distance = np.zeros((len(removing_steps)))
    for i, removing_step in enumerate(removing_steps):
        new_spike_train = spike_train_processing.remove_spikes(spike_train, removing_step)

        pif_distance[i] = spike_train_metric.cal_dist_pif(spike_train, new_spike_train, sigma)  # polarity interference measurement
        pid_distance[i] = spike_train_metric.cal_dist_pid(spike_train, new_spike_train, sigma)  # polarity independent measurement
        hamming_distance[i] = spike_train_metric.hamming_distance(spike_train, new_spike_train)  # Hamming distance measurement

    return pif_distance, pid_distance, hamming_distance
