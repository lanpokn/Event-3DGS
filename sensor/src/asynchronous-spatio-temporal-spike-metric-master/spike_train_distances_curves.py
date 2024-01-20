"""
Function: Spike train distance curves : polarity inference measurement, polarity independent measurement and Hamming distances.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Oct. 29th, 2018.

"""

import numpy as np
from matplotlib import pyplot as pl
from scipy.interpolate import spline
from event_process import generating_spike_train
from measure_spike_trains import spike_train_displacement_distances, spike_train_removing_distance


### generating spike train.
spike_train = generating_spike_train.random_array(1000000, 1000, ratio=1)


### measure distances of spike trains.
displacement_steps = np.linspace(100, 2000, 20)
pif_distance, pid_distance, hamming_distance = spike_train_displacement_distances(spike_train, displacement_steps, 1000) 

fig = pl.figure()
pl.plot(displacement_steps, hamming_distance, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='Hamming distance')
pl.plot(displacement_steps, pif_distance, '-.', color='limegreen', markersize=3, linewidth=3, figure=fig, label='Polarity inference')
pl.plot(displacement_steps, pid_distance, '-', color='red', markersize=3, linewidth=3, figure=fig, label='Polarity independent')

font1 = {'family': 'Times New Roman', 'size': 15}
font2 = {'size': 12}
pl.xlabel('Circular shift',font1)
pl.grid(axis='y', linestyle='-.')
pl.ylabel('Distortion', font1)
pl.xlim((0, 2000))
pl.ylim((0,2100))
pl.xticks(np.linspace(0, 2000, 6), fontsize=12)
pl.yticks(fontsize=12)
pl.yticks(np.linspace(0, 2000, 6), fontsize=12)
#pl.legend(loc = 0, prop=font2)
pl.legend(loc='upper center', bbox_to_anchor=(0.27, 0.92), prop=font2)
pl.show()


### show spike distances curves in removing spikes.
removing_steps = np.linspace(10, 200, 20)
pif_distance, pid_distance, hamming_distance = spike_train_removing_distance(spike_train, removing_steps, 1000) # sigma = 1000

fig = pl.figure()
new_steps = np.linspace(10, 199, 10)
smooth = spline(removing_steps, pid_distance, new_steps)

pl.plot(removing_steps, hamming_distance, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='Hamming distance')
pl.plot(removing_steps, pif_distance, '-.', color='limegreen' ,markersize=3, linewidth=3, figure=fig, label='Polarity inference')
#pl.plot(removing_steps, pid_distance, '-', color='red', markersize=3, linewidth=3, figure=fig, label='Polarity independent')
pl.plot(new_steps, smooth, '-', color='red', markersize=3, linewidth=3, figure=fig, label='Polarity independent')

font1 = {'family': 'Times New Roman', 'size': 15}
font2 = {'size': 12}
pl.xlabel('Removing spike numbers',font1)
pl.grid(axis='y', linestyle='-.')
pl.ylabel('Distortion', font1)
pl.xlim((0, 200))
pl.ylim((0,300))
pl.xticks(np.linspace(0, 200, 5), fontsize=12)
pl.yticks(fontsize=12)
pl.yticks(np.linspace(0, 300, 6), fontsize=12)
pl.legend(loc = 0, prop=font2)
pl.show()
