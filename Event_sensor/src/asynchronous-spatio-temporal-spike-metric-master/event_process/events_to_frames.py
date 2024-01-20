"""
Funciton: events to frames or videos in dynamic vision sensor(DVS).
Author information: Jianing Li, lijianing@pku.edu.cn, Peking University, May 14th, 2018.

"""

import numpy
import cv2
import numpy as np
import math


def spike_time_to_image(dvs_data, spike_time, timespan, fps, width, height):
    """
    events generate RGB video based on rate-based: spike time.

    Inputs:
    -------
        dvs_data    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        spike_time    - the maximum timestamp in events streams.
        spike_time    - the integral time interval.
        fps    - the frames per second.
        width    - the width of AER sensor.
        height    - the height of AER sensor.

    Outputs:
    ------
        video     - video includes multi-frames.

    """
    spike_frame = numpy.zeros([height, width, 3])  # RGB background---black
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoWriter = cv2.VideoWriter('timesVideo.avi', fourcc, fps, (width, height), True)  # DAVIS128 Width*Height:128*128

    """Read while aedat file: spike map in RGB frame, data.x is x coordinate, data.y is y coordinate"""
    for i in range(0, spike_time, timespan):
        for j in range(len(dvs_data.t)):
            if dvs_data.ts[j] >= i and dvs_data.ts[j] <= i+timespan:
                if dvs_data.t[j] == 1:
                    spike_frame[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1), :] = [0, 0, 255]  # positive---read

                if dvs_data.t[j] == 0:
                    spike_frame[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1), :] = [0, 255, 0]  # negative---green

        spike_frame = spike_frame.astype('uint8')  # float to uint8
        VideoWriter.write(spike_frame)

        cv2.imwrite('upsampling.jpg',spike_frame)

    VideoWriter.release()

def spikes_to_images(dvs_data, width, height):
    """Function: spike in timespan to reconstuct three RGB video"""
    spike_frame_on = np.zeros([height, width, 3])  # RGB background---black
    spike_frame_off = np.zeros([height, width, 3])
    spike_frame = np.zeros([height, width, 3])
    spike_number_on = np.zeros((height, width))  # spike frame initial in time interval
    spike_number_off = np.zeros((height, width))
    spike_on = np.zeros((height, width))  # single chanel information
    spike_off = np.zeros((height, width))
    spike_number = np.zeros((height, width))

    """Read while aedat file: spike map in RGB frame, data.x is x coordinate, data.y is y coordinate"""
    for j in range(len(dvs_data.t)):
        spike_number[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1)] += 1
        if dvs_data.t[j] == 1:
            spike_number_on[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1)] += 1
            spike_frame_on[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1), :] = [0, 0, 255]  # positive---read
            spike_on[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1)] = 255

        if dvs_data.t[j] == 0:
            spike_number_off[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1)] += 1
            spike_frame_off[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1), :] = [0, 255, 0]  # negative---green
            spike_off[int(dvs_data.x[j] - 1), int(dvs_data.y[j] - 1)] = 255

    for n in range(width):
        for m in range(height):
            if spike_number_on[m - 1, n - 1] + spike_number_off[
                m - 1, n - 1] == 0:  # threshold can be used as the filter
                spike_frame[m - 1, n - 1, :] = [255, 255, 255]  # gray backgroud
                # spike_frame[m-1, n-1, :] = [0, 0, 0] #gray backgroud

            else:
                if spike_number_on[m - 1, n - 1] >= spike_number_off[m, n]:
                    spike_frame[m - 1, n - 1, :] = [0, 0, 255]  # positive---read

                else:
                    spike_frame[m - 1, n - 1, :] = [0, 255, 0]  # negative---green

    spike_count = spike_number.reshape(width * height, 1)
    for k in range(len(spike_count) - 1):
        if spike_count[k] == 0:
            spike_count[k] = 128
        else:
            spike_count[k] = 255 * (1 / (1 + math.exp(-1 / 2 * spike_count[k])))
    gray_frame = spike_count.reshape(height, width)
    gray_frame = gray_frame.astype('uint8')  # float to uint8


    return spike_frame