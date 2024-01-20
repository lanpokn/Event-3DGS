"""
Function: Show measure distances of spike cubes.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Jan. 13th, 2019.

"""

import numpy as np
from event_process import read_dvs
from event_process import event_processing
from measure_spike_cubes import spike_cubes_rotation_distances, spike_cubes_translation_distances, spike_cubes_random_change_distances, spike_cubes_removing_distances, spike_cubes_displacement_distances, polarity_percentage_chaning_distances, increasing_noise_distances
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from event_process.show_events import  show_ON_OFF_events, show_ON_OFF_spikes, show_davis240_ON_OFF_spikes, show_simulating_events
from scipy.io import loadmat
import math

### read event-based data for DVS128.
base_dir = './datasets/dvs_mnist/'
filename = base_dir + 'mnist_7_scale08_0001_QS1.aedat'
aer_file = read_dvs.aefile(filename)
aer_data = read_dvs.aedata(aer_file)
events = event_processing.aer_events(aer_data)

#show_ON_OFF_events(events, width = 128, height =128, length = max(events[1, :]))

# # read event-based data for DVS.
# base_filename = './testing_scalability/metric_rec_dvs/bicycle/bicycle_s1_0.2.mat'
# events_mat = loadmat(base_filename)
# base_events = events_mat["new_events"]
# base_events[0, :] = base_events[0, :] - base_events[0, 0]
# for h in range(base_events.shape[1]):
#     if base_events[3, h] == 0:
#         base_events[3, h] = -1

# new_events = np.copy(base_events)
# base_events[0, :] = new_events[3, :]
# base_events[1, :] = new_events[0, :]
# #base_events[2, :] = 240 - new_events[2, :]-1
# base_events[3, :] = new_events[1, :]

# show_davis240_ON_OFF_spikes(base_events, 240, 180)


# pkl_events = open('./datasets/simulating_dataset/events_3d_trigonometric.pkl', 'rb')
# #pkl_events = open('./datasets/simulating_dataset/events_2d_trigonometric.pkl', 'rb')
# events = pickle.load(pkl_events)
# show_simulating_events(events, width=128, height=128)

def show_spike_cubes_rotation(events, max_rotation, width, height):
    """
    show spike cubes distances in rotation operation.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_rotation    - the spike cube rotation range(0-360).

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    rotation_steps = np.linspace(0, max_rotation, 21)  # 21 removing spike steps
    kernel_train_distances, kernel_cube_distances = spike_cubes_rotation_distances(events, rotation_steps, width=width, height=height)

    #pickle.dump(rotation_steps, open('./testing_supplementary/cube_parameters/rotation_steps.pkl', 'wb'))
    #pickle.dump(kernel_cube_distances, open('./testing_supplementary/cube_parameters/temporal_1200.pkl', 'wb'))

    # show distances curve.
    fig = plt.figure()
    # plt.plot(removing_spike_steps, fft_l1_distances, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='3D convolution')
    # plt.plot(removing_spike_steps, kernel_train_distances, '-.', color='limegreen', markersize=3, linewidth=3, figure=fig, label='Spike train kernel')
    plt.plot(rotation_steps, kernel_cube_distances, '-', color='red', markersize=3, linewidth=3, figure=fig, label='3D gaussian kernel')

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = {'size': 16}
    # plt.xlabel('Removing spike numbers', font1)
    plt.xlabel(r'$R_\theta$', font1)
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('Distance', font1)
    plt.xlim((0, max_rotation))
    plt.ylim((0, 5250))
    plt.xticks(np.linspace(0, max_rotation, 3), fontproperties = 'Times New Roman', fontsize=18)
    plt.yticks(np.linspace(0, 5250, 4), fontproperties = 'Times New Roman', fontsize=18)
    plt.show()


def show_temporal_spatial_translation(events, max_width, max_height):
    """
    show spike cubes distances in temporal and spatial random changing steps.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_width    - the maximum spatial changing size.
        max_height   - the maximum temporal changing size.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """
    temporal_width_steps = np.linspace(0, max_width, 6)  # 21 temporal random changing steps
    spatial_height_steps = np.linspace(0, max_height, 6)  # 6 spatial random changing steps
    kernel_train_distances, kernel_cube_distances = spike_cubes_translation_distances(events, temporal_width_steps, spatial_height_steps, width=128, height=128)

    new_cube_distances = np.copy(kernel_cube_distances)
    for i in range(len(kernel_cube_distances[0, :])):
        new_cube_distances[:, i] = np.sort(kernel_cube_distances[:, i])

    #for i in range(kernel_cube_distances[0, :])：
    # show distances matrix curve surface.
    fig = plt.figure()
    ax = Axes3D(fig)
    temporal_width_steps, spatial_height_steps = np.meshgrid(temporal_width_steps, spatial_height_steps)
    ax.plot_surface(temporal_width_steps, spatial_height_steps, new_cube_distances.transpose(), rstride=1, cstride=1, cmap='rainbow')

    font1 = {'family': 'Times New Roman', 'size': 21}
    font2 = {'family': 'Times New Roman', 'size': 18}
    ax.set_xlabel(r'$\phi_X$', font1)
    ax.set_ylabel(r'$\phi_Y$', font1)
    ax.set_zlabel('Distance', font1)
    ax.set_zticks(np.linspace(0, 25000, 3).astype(int))
    ax.set_xticks(np.linspace(0, max_width, 3).astype(int))
    ax.set_yticks(np.linspace(0, max_height, 3).astype(int))
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")
    ax.zaxis.set_tick_params(labelsize=18)
    ax.tick_params(axis='x', which='major', pad=-2)
    ax.tick_params(axis='y', which='major', pad=-2)
    ax.tick_params(axis='z', which='major', pad=0)
    # ax.set_xticks(np.linspace(0, 8, 5), fontsize=14)
    # plt.xticks(np.linspace(0, max_temporal_changing, 6), fontsize=14)
    # plt.yticks(np.linspace(0, max_spatial_changing, 6), fontsize=14)
    # plt.xticks(np.linspace(0, 8, 3), fontsize=16)
    # plt.yticks(np.linspace(0, 20, 3), fontsize=16)

    plt.show()


def show_temporal_spatial_random_changing(events, max_temporal_changing, max_spatial_changing):
    """
    show spike cubes distances in temporal and spatial random changing steps.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_temporal_changing    - the maximum temporal random changing steps.
        max_spatial_changing    - the maximum spatial random changing steps.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    temporal_random_steps = np.linspace(0, max_temporal_changing, 21)  # 21 temporal random changing steps
    spatial_random_steps = np.linspace(0, max_spatial_changing, 6)  # 6 spatial random changing steps
    fft_l1_distances, kernel_train_distances, kernel_cube_distances = spike_cubes_random_change_distances(events, temporal_random_steps, spatial_random_steps, width=128, height=128)

    # save the distance in pkl.
    pickle.dump(fft_l1_distances, open('./datasets/temporal_spatial_changing/fft_l1_distances.pkl', 'wb'))
    pickle.dump(kernel_train_distances, open('./datasets/temporal_spatial_changing/kernel_train_distances.pkl', 'wb'))
    pickle.dump(kernel_cube_distances, open('./datasets/temporal_spatial_changing/kernel_cube_distances.pkl', 'wb'))

    # show distances matrix curve surface.
    fig = plt.figure()
    ax = Axes3D(fig)
    temporal_random_steps, spatial_random_steps = np.meshgrid(temporal_random_steps, spatial_random_steps)
    ax.plot_surface(temporal_random_steps/1000, spatial_random_steps, kernel_cube_distances.transpose(), rstride=1, cstride=1, cmap='rainbow')

    font1 = {'family': 'Times New Roman', 'size': 21}
    font2 = {'family': 'Times New Roman', 'size': 18}
    ax.set_xlabel(r'$\psi_T$', font1)
    ax.set_ylabel(r'$\psi_S$', font1)
    ax.set_zlabel('Distance', font1)
    ax.set_zticks(np.linspace(0, 8000, 3).astype(int))
    ax.set_xticks(np.linspace(0, 8, 3).astype(int))
    ax.set_yticks(np.linspace(0, 20, 3).astype(int))
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")
    ax.zaxis.set_tick_params(labelsize=18)
    ax.tick_params(axis='x', which='major', pad=-2)
    ax.tick_params(axis='y', which='major', pad=-2)
    ax.tick_params(axis='z', which='major', pad=0)
    #ax.set_xticks(np.linspace(0, 8, 5), fontsize=14)
    #plt.xticks(np.linspace(0, max_temporal_changing, 6), fontsize=14)
    #plt.yticks(np.linspace(0, max_spatial_changing, 6), fontsize=14)
    # plt.xticks(np.linspace(0, 8, 3), fontsize=16)
    # plt.yticks(np.linspace(0, 20, 3), fontsize=16)


    plt.show()


def show_random_removing_spikes(events, max_removing_numbers):
    """
    show spike cubes distances in random removing spike numbers.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_removing_numbers    - the maximum random  removing spike numbers.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    removing_spike_steps = np.linspace(0, max_removing_numbers, 21)  # 21 removing spike steps
    fft_l1_distances, kernel_train_distances, kernel_cube_distances = spike_cubes_removing_distances(events, removing_spike_steps, width=128, height=128)

    # save the distance in pkl.
    pickle.dump(fft_l1_distances, open('./datasets/random_removing_spikes/fft_l1_distances.pkl', 'wb'))
    pickle.dump(kernel_train_distances, open('./datasets/random_removing_spikes/kernel_train_distances.pkl', 'wb'))
    pickle.dump(kernel_cube_distances, open('./datasets/random_removing_spikes/kernel_cube_distances.pkl', 'wb'))

    # show distances curve.
    fig = plt.figure()
    # plt.plot(removing_spike_steps, fft_l1_distances, '--', color='blue', markersize=3, linewidth=3, figure=fig, label='3D convolution')
    # plt.plot(removing_spike_steps, kernel_train_distances, '-.', color='limegreen', markersize=3, linewidth=3, figure=fig, label='Spike train kernel')
    plt.plot(removing_spike_steps, kernel_cube_distances, '-', color='red', markersize=3, linewidth=3, figure=fig, label='3D gaussian kernel')

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = {'size': 16}
    # plt.xlabel('Removing spike numbers', font1)
    plt.xlabel(r'$N$', font1)
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('Distance', font1)
    plt.xlim((0, max_removing_numbers))
    plt.ylim((0, 48000))
    plt.xticks(np.linspace(0, max_removing_numbers, 3), fontproperties = 'Times New Roman', fontsize=18)
    plt.yticks(np.linspace(0, 48000, 4), fontproperties = 'Times New Roman', fontsize=18)
    plt.show()



def show_cyclic_displacement(events, max_temporal_cyclic, max_spatial_cyclic):
    """
    show spike cubes distances in temporal and spatial cyclic displacement steps.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_temporal_cyclic    - the maximum temporal cyclic displacement changing steps.
        max_spatial_cyclic    - the maximum temporal cyclic displacement changing steps.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    temporal_cyclic_displacement_steps = np.linspace(0, max_temporal_cyclic, 11)  # 11 temporal cyclic displacement steps
    spatial_cyclic_displacement_steps = np.linspace(0, max_spatial_cyclic, 6)  # 6 spatial cyclic displacement steps
    fft_l1_distances, kernel_train_distances, kernel_cube_distances = spike_cubes_displacement_distances(events, temporal_cyclic_displacement_steps, spatial_cyclic_displacement_steps, width=128, height=128)

    # save the distance in pkl.
    pickle.dump(fft_l1_distances, open('./datasets/cyclic_displacement/fft_l1_distances.pkl', 'wb'))
    pickle.dump(kernel_train_distances, open('./datasets/cyclic_displacement/kernel_train_distances.pkl', 'wb'))
    pickle.dump(kernel_cube_distances, open('./datasets/cyclic_displacement/kernel_cube_distances.pkl', 'wb'))

    # show distances in 3D surface.
    fig = plt.figure()
    ax = Axes3D(fig)

    temporal_cyclic_displacement_steps, spatial_cyclic_displacement_steps = np.meshgrid(temporal_cyclic_displacement_steps, spatial_cyclic_displacement_steps)
    ax.plot_surface(temporal_cyclic_displacement_steps/1000, spatial_cyclic_displacement_steps, kernel_cube_distances.transpose(), rstride=1, cstride=1, cmap='rainbow')

    font1 = {'family': 'Times New Roman', 'size': 21}
    font2 = {'family': 'Times New Roman', 'size': 18}
    ax.set_xlabel(r'$\phi_T$', font1)
    ax.set_ylabel(r'$\phi_S$', font1)
    ax.set_zlabel('Distance', font1)
    ax.set_zticks(np.linspace(0, 20000, 3).astype(int))
    ax.set_xticks(np.linspace(0, 4, 3).astype(int))
    ax.set_yticks(np.linspace(0, 10, 3).astype(int))
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")
    ax.zaxis.set_tick_params(labelsize=18)
    ax.tick_params(axis='x', which='major', pad=-2)
    ax.tick_params(axis='y', which='major', pad=-2)
    ax.tick_params(axis='z', which='major', pad=-2)
    # ax.set_xticks(np.linspace(0, 8, 5), fontsize=14)
    # plt.xticks(np.linspace(0, max_temporal_changing, 6), fontsize=14)
    # plt.yticks(np.linspace(0, max_spatial_changing, 6), fontsize=14)

    plt.show()



def show_changing_polarity_percentage(events, max_changing_percentage):
    """
    show spike cubes distances in various changing percentages of ON/OFF polarity.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_changing_percentage    - the maximum changing percentage of ON/OFF polarity step.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    random_changing_percentage_steps = np.linspace(0, max_changing_percentage, 11)  # 11 random percentage

    fft_l1_distances, kernel_train_distances, kernel_cube_distances = polarity_percentage_chaning_distances(events, random_changing_percentage_steps)

    # save the distance in pkl.
    #pickle.dump(random_changing_percentage_steps, open('./datasets/changing_polarity/polarity_changing_steps.pkl', 'wb'))
    #pickle.dump(fft_l1_distances, open('./datasets/cyclic_displacement/fft_l1_distances.pkl', 'wb'))
    #pickle.dump(kernel_train_distances, open('./datasets/changing_polarity/kernel_train_pid_distances.pkl', 'wb'))
    #pickle.dump(kernel_cube_distances, open('./datasets/changing_polarity/kernel_cube_distances.pkl', 'wb'))

    fig = plt.figure()
    plt.plot(random_changing_percentage_steps, kernel_cube_distances, '-', color='red', markersize=3, linewidth=3, figure=fig, label='3D gaussian kernel')
    # plt.plot(random_changing_percentage_steps, kernel_train_distances, '-', color='green', markersize=3, linewidth=3, figure=fig, label='spike train')

    font1 = {'family': 'Times New Roman', 'size': 15}
    plt.xlabel('Polarity changing percentage', font1)
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('Distortion', font1)
    plt.xlim((0, max_changing_percentage))
    plt.xticks(np.linspace(0, max_changing_percentage, 6), fontsize=12)
    plt.show()


def show_increasing_noise_ratio(events, max_noise_ratio):
    """
    show spike cubes distances in various changing noise ratios.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        max_noise_ratio    - the maximum noise ratio, namely noise:signal.

    Outputs:
    -------
        distances figures    - the distance figures of spike cubes after random changing based on 3d gaussian kernel method.

    """

    noise_ratio_steps = np.linspace(0, max_noise_ratio, 11)  # 11 random percentage

    fft_l1_distances, kernel_train_distances, kernel_cube_distances = increasing_noise_distances(events, noise_ratio_steps)

    # save the distance in pkl.
    pickle.dump(fft_l1_distances, open('./datasets/increasing_noise/fft_l1_distances.pkl', 'wb'))
    pickle.dump(kernel_train_distances, open('./datasets/increasing_noise/kernel_train_distances.pkl', 'wb'))
    pickle.dump(kernel_cube_distances, open('./datasets/increasing_noise/kernel_cube_distances.pkl', 'wb'))

    fig = plt.figure()
    plt.plot(noise_ratio_steps, kernel_cube_distances, '-', color='red', markersize=3, linewidth=3, figure=fig, label='3D gaussian kernel')

    font1 = {'family': 'Times New Roman', 'size': 15}
    plt.xlabel('Noise ratio', font1)
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('Distortion', font1)
    plt.xlim((0, max_noise_ratio))
    plt.xticks(np.linspace(0, max_noise_ratio, 6), fontsize=12)
    plt.show()


# show_temporal_spatial_random_changing(events, 8000, 20)
# show_random_removing_spikes(events, len(events[2])-2000)
# show_cyclic_displacement(events, 4000, 10)
# show_changing_polarity_percentage(events, 0.9)
# show_increasing_noise_ratio(events, 1)
# show_temporal_spatial_translation(events,50, 50)
show_spike_cubes_rotation(events, 360, width=128, height=128)

