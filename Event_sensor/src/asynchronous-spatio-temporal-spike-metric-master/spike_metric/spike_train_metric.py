"""
Function: Inner product metric in a representation Hilbert space (RKHS).
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Oct. 28th, 2018.
Code source: the main is coded by Zhichao Bi, Peking University.
"""

import numpy as np
import math

def cal_ip_pif(array1, array2, sigma):
    tss1 = np.nonzero(array1)[0]
    tss2 = np.nonzero(array2)[0]

    ps1 = array1[tss1]
    ps2 = array2[tss2]
    dist_matrix = np.exp(-(tss1[:, None] - tss2[None, :]) ** 2 / (2 * sigma ** 2))
    dist_matrix *= ps1[:, None] * ps2[None, :]

    return np.sum(dist_matrix)

def cal_ip_pid(array1, array2, sigma):
    tss1 = np.nonzero(array1)[0]
    tss2 = np.nonzero(array2)[0]

    dist_matrix = np.exp(-(tss1[:, None] - tss2[None, :]) ** 2 / (2 * sigma ** 2))
    total_dist = np.sum(dist_matrix)
    return total_dist

def cal_scaling_ip_pid(array1, array2, sigma):
    tss1 = np.nonzero(array1)[0]
    tss2 = np.nonzero(array2)[0]
    ps1 = array1[tss1]
    ps2 = array2[tss2]
    dist_matrix = np.exp(-(tss1[:, None] - tss2[None, :]) ** 2 / (2 * sigma ** 2))
    total_dist = np.sum(dist_matrix)
    count1_on = np.count_nonzero(ps1 == 1) / max(np.count_nonzero(ps1), 1)
    count2_on = np.count_nonzero(ps2 == 1) / max(np.count_nonzero(ps2), 1)
    scaling = count1_on * count2_on + (1 - count1_on) * (1 - count2_on)
    # scaling = 1 + abs(count1_on - count2_on)
    return total_dist * scaling


def cal_dist_pif(array1, array2, sigma):
    total_dist = cal_ip_pif(array1, array1, sigma=sigma)
    total_dist += cal_ip_pif(array2, array2, sigma=sigma)
    total_dist -= 2 * cal_ip_pif(array1, array2, sigma=sigma)

    return total_dist


def cal_dist_pid(array1, array2, sigma):
    total_dist = cal_ip_pid(array1, array1, sigma=sigma) # no considering polarity.
    total_dist += cal_ip_pid(array2, array2, sigma=sigma)
    total_dist -= 2 * cal_ip_pid(array1, array2, sigma=sigma)

    # total_dist = cal_scaling_ip_pid(array1, array1, sigma=sigma) # considering polarity probability in point process history.
    # total_dist += cal_scaling_ip_pid(array2, array2, sigma=sigma)
    # total_dist -= 2 * cal_scaling_ip_pid(array1, array2, sigma=sigma)

    return total_dist


def hamming_distance(array1, array2):
    delta_array = np.nonzero(np.array(array1) - np.array(array2))
    distance = len(delta_array[0])

    return distance