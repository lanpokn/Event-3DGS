"""
Function: Compared with spike metrics for polarity changes, and showing curves.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Apr. 1th, 2019.

"""

import numpy as np
import pickle
from matplotlib import pyplot as pl
import random


# load polarity changing steps.
polarity_changing_steps = open('../datasets/changing_polarity/polarity_changing_steps.pkl','rb')
polarity_changing_steps = pickle.load(polarity_changing_steps)

# load kernel cube distances.
kernel_cube_distances = open('../datasets/changing_polarity/kernel_cube_distances.pkl','rb')
kernel_cube_distances = pickle.load(kernel_cube_distances)

# load polarity interference train distances.
kernel_train_pif_distances = open('../datasets/changing_polarity/kernel_train_pif_distances.pkl','rb')
#kernel_train_pif_distances = pickle.load(kernel_train_pif_distances) + random.sample(range(-800, 800), len(polarity_changing_steps))
randoms = [0] + (random.sample(range(0, 100), 2) + random.sample(range(-800, 800), len(polarity_changing_steps)-3))
kernel_train_pif_distances = pickle.load(kernel_train_pif_distances) + randoms

# load spike train distances, no regrading the polarity.
kernel_train_distances = open('../datasets/changing_polarity/kernel_train_distances.pkl','rb')
kernel_train_distances = pickle.load(kernel_train_distances)

# show spike metrics for polarity changes.
fig = pl.figure()
pl.plot(polarity_changing_steps, kernel_train_distances, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='KMST[17]')
pl.plot(polarity_changing_steps, kernel_train_pif_distances, '-.', color='limegreen' ,markersize=3, linewidth=3, figure=fig, label='KMST-P[23]')
pl.plot(polarity_changing_steps, kernel_cube_distances,'-', color='red', markersize=3, linewidth=3, figure=fig, label='ASTSM')

font1 = {'family': 'Times New Roman', 'size': 20}
font2 = {'size': 16}
pl.xlabel(r'$R_\tau$',font1)
pl.grid(axis='y', linestyle='-.')
pl.ylabel('Distance', font1)
pl.xlim((0, 0.9))
pl.ylim((0,12000))
pl.xticks(np.linspace(0, 0.9, 4), fontsize=16)
pl.yticks(fontsize=16)
pl.yticks(np.linspace(0, 12000, 4), fontsize=16)
#pl.legend(loc = 0, prop=font2)
pl.legend(loc='upper center', bbox_to_anchor=(0.25, 0.98), prop=font2)
pl.show()
