"""
Function: spike train processing: displacement and removing.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Oct. 29th, 2018.

"""

import numpy as np
import random


def displacement_spikes(spike_train, k):
    """
    cyclic displacement for spike train.

    Inputs:
    -------
        spike_train    - numpy.array: the spike train includes spiking firing timestamp.
        k    - the cyclic displacement number.

    Outputs:
    -------
        new_spike_train    - the new spike train by cyclic displacement.

    """

    spike_train = spike_train.tolist()

    new_spike_train = spike_train[int(k):] + spike_train[:int(k)]
    new_spike_train = np.array(new_spike_train)

    return new_spike_train

# new_A = displacement_spikes(A, 2)


def remove_spikes(spike_train, spike_numbers):
    """
        remove spike numbers for spike train.

        Inputs:
        -------
            spike_train    - numpy.array: the spike train includes spiking firing timestamp.
            spike_numbers    - the removing spike numbers.

        Outputs:
        -------
            new_spike_train    - the new spike train by removing spikes.

    """
    new_spike_train = np.copy(spike_train)
    indexs = np.nonzero(spike_train)[0]

    remove_indexs = random.sample(range(0, len(indexs)), int(spike_numbers))

    for i, remove_index in enumerate(remove_indexs):
        new_spike_train[indexs[remove_index]] = 0

    return new_spike_train

