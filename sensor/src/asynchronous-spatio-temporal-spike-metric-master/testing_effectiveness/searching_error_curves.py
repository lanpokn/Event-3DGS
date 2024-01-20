"""
Function: Show searching error curves under various distortion and compared approaches.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Mar. 5th, 2019.

"""
import numpy as np
import pickle
from testing_effectiveness.searching_trigonometric_target import search_trigonometric_trajectory, computing_trajectory_error
from event_process.spike_cube_processing import events_increasing_noises
from event_process.show_events import  show_simulating_events

def show_searching_error_increasing_noise(events, true_target, max_noise_ratio):
    """
    Show searching trajectory errors under increasing random noise.

    Inputs:
    -------
        events   - the dynamic vision sensor.
        true_target    - the ground truth of moving trajectory, using numpy matrix.
        max_noise_ratio    - the maximum ratio of noise to signal.

    Outputs:
    ------
        figure     - a figure shows searching trajectory errors.

    """

    noise_ratio_steps = np.linspace(0, max_noise_ratio, 11)  # 11 random percentage

    # pickle.dump(noise_ratio_steps, open('../datasets/search_moving_target/noise_ratio_steps.pkl', 'wb'))

    mean_errors = np.zeros((len(noise_ratio_steps)))

    for i, noise_ratio_step in enumerate(noise_ratio_steps):

        new_events = events_increasing_noises(events, noise_ratio_step, 1, 128, 128)

        # show_simulating_events(new_events, width=128, height=128, length=max(new_events[1, :]))


        search_centers = search_trigonometric_trajectory(true_target, new_events, 10, 10, 200, 16, 16) # 2d - 8, , 3d - 16, 16 search areas

        trajectory_errors, mean_errors[i] = computing_trajectory_error(true_target, search_centers)

        print('noise_ratio_step = {}, mean_error = {}'.format(noise_ratio_step, mean_errors[i]))

    return mean_errors



if __name__ == '__main__':

    # read simulating data.
    pkl_events = open('../datasets/simulating_dataset/events_3d_trigonometric.pkl', 'rb')
    pkl_true_target = open('../datasets/simulating_dataset/events_3d_true_target.pkl', 'rb')
    events = pickle.load(pkl_events)
    true_target = pickle.load(pkl_true_target)

    mean_errors = show_searching_error_increasing_noise(events, true_target, 2) # maximum ratio of noise to signal.

    # save searching errors in pkl.
    # pickle.dump(mean_errors, open('../datasets/search_moving_target/kernel_cube_errors.pkl', 'wb'))
    # pickle.dump(mean_errors, open('../datasets/search_moving_target/kernel_train_pif_errors.pkl', 'wb'))


    print('pku')