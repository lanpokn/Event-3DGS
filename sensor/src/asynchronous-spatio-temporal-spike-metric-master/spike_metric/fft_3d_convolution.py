"""
Function: Spike train distance curves : polarity inference measurement, polarity independent measurement and Hamming distances.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Jan. 22th, 2019.

"""

import numpy as np
import math

def get_3d_gaussian_kernel(x_size=10, y_size=10, t_size=1000, x_sigma=10, y_sigma=10, t_sigma=500):
    """
    get a 3d gaussian kernel.

    Inputs:
    -------
        x_size    - the size of 3d gaussian kernel in x dimension.
        y_size    - the size of 3d gaussian kernel in y dimension.
        t_size    - the size of 3d gaussian kernel in t dimension.
        x_sigma   - the x parameter of 3d gaussian kernel.
        y_sigma   - the y parameter of 3d gaussian kernel.
        t_sigma   - the z parameter of 3d gaussian kernel.
    Outputs:
    -------
        gaussian_3d    - the 3d gaussian kernel in discrete type.

    """

    x_vec = np.arange(-math.floor(x_size/2), math.floor(x_size/2), 1)
    y_vec = np.arange(-math.floor(y_size/2), math.floor(y_size/2), 1)
    t_vec = np.arange(-math.floor(t_size/2), math.floor(t_size/2), 1)

    xx, yy, tt = np.meshgrid(x_vec, y_vec, t_vec)

    gauss_3d = np.exp(-xx ** 2 / 2 * x_sigma ** 2 - yy ** 2 / 2 * y_sigma ** 2 - tt ** 2 / 2 * t_sigma ** 2)

    return gauss_3d/np.sum(gauss_3d)


def fft_convolution(spike_cube, gaussian_3d):
    """
    3d convolution based on fft (fast fourier transform).

    Inputs:
    -------
        spike_cube    - the spike array.
        gaussian_3d    - the 3d gaussian convolution kernel.
    Outputs:
    -------
        inverse_fft    - the inverse fft to compute 3d convolution.

    """

    fft1 = np.fft.fftn(gaussian_3d, spike_cube.shape)
    fft2 = np.fft.fftn(spike_cube)
    inverse_fft = np.real(np.fft.ifftn(fft1*fft2))

    return inverse_fft




