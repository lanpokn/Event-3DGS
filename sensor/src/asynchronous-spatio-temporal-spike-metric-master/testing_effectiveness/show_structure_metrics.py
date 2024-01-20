"""
Function: Show searching error curves for spike metrics.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Apr. 1st, 2019.

"""

import numpy as np
import pickle
from matplotlib import pyplot as pl

# load noise ratio steps.
noise_ratio_steps = open('../datasets/search_moving_target/noise_ratio_steps.pkl','rb')
noise_ratio_steps = pickle.load(noise_ratio_steps)

# load kernel cube errors.
kernel_cube_errors = open('../datasets/search_moving_target/kernel_cube_errors.pkl','rb')
kernel_cube_errors = pickle.load(kernel_cube_errors)
kernel_cube_errors = kernel_cube_errors[np.argsort(kernel_cube_errors)]

# load kernel train errors.
kernel_train_errors = open('../datasets/search_moving_target/kernel_train_errors.pkl','rb')
kernel_train_errors = pickle.load(kernel_train_errors)
kernel_train_errors = kernel_train_errors[np.argsort(kernel_train_errors)]/2 # show better curve

# load kernel train errors using polarity interference.
kernel_train_pif_errors = open('../datasets/search_moving_target/kernel_train_pif_errors.pkl','rb')
kernel_train_pif_errors = pickle.load(kernel_train_pif_errors)
kernel_train_pif_errors = kernel_train_pif_errors[np.argsort(kernel_train_pif_errors)]/2 # show better curve

# show spike metrics for structure attribute.
fig = pl.figure()
pl.plot(noise_ratio_steps, kernel_train_errors, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='KMST[17]')
pl.plot(noise_ratio_steps, kernel_train_pif_errors, '-.', color='limegreen' ,markersize=3, linewidth=3, figure=fig, label='KMST-P[23]')
pl.plot(noise_ratio_steps, kernel_cube_errors,'-', color='red', markersize=3, linewidth=3, figure=fig, label='ASTSM')

font1 = {'family': 'Times New Roman', 'size': 20}
font2 = {'size': 16}
pl.xlabel(r'$N_\tau$',font1)
pl.grid(axis='y', linestyle='-.')
pl.ylabel('Tracking errors / pixel', font1)
pl.xlim((0, 2))
pl.ylim((0, 60))
pl.xticks(np.linspace(0, 2, 5), fontsize=16)
pl.yticks(fontsize=16)
pl.yticks(np.linspace(0, 60, 5), fontsize=16)
#pl.legend(loc = 0, prop=font2)
pl.legend(loc='upper center', bbox_to_anchor=(0.24, 0.98), prop=font2)
pl.show()

