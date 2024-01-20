"""
Function: Random generating spike trains.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Oct. 29th, 2018.

"""

import numpy as np
import random

def random_array(total_numbers, spike_numbers, ratio = 2):
    """
    random generating spike train for metric.

    Inputs:
    -------
        total_numbers    - the total numbers of timestamp in spike train.
        spike_numbers    - the spike numbers.
        ratio    - ON/OFF=2.

    Outputs:
    -------
        spike_train    - the random generating spike train.

    """
    samples = random.sample(range(0, total_numbers), spike_numbers)
    spike_train = np.zeros((total_numbers))

    for i, index in enumerate(samples):
        if i < len(samples)/(ratio+1):
            spike_train[index] = -1
        else:
            spike_train[index] = 1

    return spike_train