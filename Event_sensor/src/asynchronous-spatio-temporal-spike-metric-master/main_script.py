"""
Measure distances between spike cubes using various distortion operations.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Jan. 13th, 2019.

"""

import os
import argparse
from event_process import read_dvs
from event_process import event_processing
import spike_cubes_distances_curves


argparser = argparse.ArgumentParser(
    description='Measure distances between spike cubes using various distortion operations.'
    )

argparser.add_argument(
    '-d',
    '--data_path',
    help='The .aedat data path using dynamic vision sensor',
    default='./datasets/mnist_0_scale04_0001.aedat'
    )

argparser.add_argument(
    '-o',
    '--operation_type',
    help='The distortion operation to spike cube, such as spatial_temporal_changing, changing polarity, cyclic_displacement, increasing noise, random_removing_spikes...',
    default='spatial_temporal_changing'
    )


def _main(args):

    ### read event-based data.
    data_path = os.path.expanduser(args.data_path)
    aer_file = read_dvs.aefile(data_path)
    aer_data = read_dvs.aedata(aer_file)
    events = event_processing.aer_events(aer_data)

    ### show distances under various distortion operations.
    if args.operation_type == "spatial_temporal_changing":
        spike_cubes_distances_curves.show_temporal_spatial_random_changing(events, 8000, 10)

    elif args.operation_type == "temporal_spatial_translation":
        spike_cubes_distances_curves.show_temporal_spatial_translation(events,50, 50)

    elif args.operation_type == "spike_cube_rotation":
        spike_cubes_distances_curves.show_spike_cubes_rotation(events, 360, width=128, height=128)

    elif args.operation_type == "removing_spikes":
        spike_cubes_distances_curves.show_random_removing_spikes(events, 1000)

    elif args.operation_type  == "cylic_displacement":
        spike_cubes_distances_curves.show_cyclic_displacement(events, 4000, 5)

    elif args.operation_type == "changing_polarity":
        spike_cubes_distances_curves.show_changing_polarity_percentage(events, 0.9)

    elif args.operation_type == "increasing noise":
        spike_cubes_distances_curves.show_increasing_noise_ratio(events, 1)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)

