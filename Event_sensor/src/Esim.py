# edit from ESIM
from numba.np.ufunc import parallel
import numpy as np
from types import SimpleNamespace
from numba import njit, prange, set_num_threads
import cv2
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import argparse
import sys, signal
import pandas as pd
import pickle
from event_buffer import EventBuffer
from event_file_io import EventsData
from event_display import EventDisplay

EVENT_TYPE = np.dtype(
    [("timestamp", "f8"), ("x", "u2"), ("y", "u2"), ("polarity", "b")], align=True
)

TOL = 0.5
MINIMUM_CONTRAST_THRESHOLD = 0.01

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_ns": 1000,
        "max_events_per_frame": 200000,
    }
)

def esim(
    x_end,
    current_image,
    previous_image,
    delta_time,
    crossings,
    last_time,
    output_events,
    spikes,
    refractory_period_ns,
    max_events_per_frame,
    n_pix_row,
):
    count = 0
    max_spikes = int(delta_time / (refractory_period_ns * 1e-3))
    for x in prange(x_end):
        itdt = np.log(current_image[x])
        it = np.log(previous_image[x])
        deltaL = itdt - it

        if np.abs(deltaL) < TOL:
            continue

        pol = np.sign(deltaL)

        cross_update = pol * TOL
        crossings[x] = np.log(crossings[x]) + cross_update

        lb = crossings[x] - it
        ub = crossings[x] - itdt

        pos_check = lb > 0 and (pol == 1) and ub < 0
        neg_check = lb < 0 and (pol == -1) and ub > 0

        spike_nums = (itdt - crossings[x]) / TOL
        cross_check = pos_check + neg_check
        spike_nums = np.abs(int(spike_nums * cross_check))

        crossings[x] = itdt - cross_update
        if spike_nums > 0:
            spikes[x] = pol

        spike_nums = max_spikes if spike_nums > max_spikes else spike_nums

        current_time = last_time
        for i in range(spike_nums):
            output_events["x"][count] = x % n_pix_row
            output_events["y"][count] = x // n_pix_row
            output_events["timestamp"][count] = current_time
            output_events["polarity"][count] = 1 if pol > 0 else 0

            count += 1
            current_time += (delta_time) / spike_nums

            if count == max_events_per_frame:
                return count

    return count


class EventSimulator:
    def __init__(self, W, H, first_image=None, first_time=None, config=CONFIG):
        self.H = H
        self.W = W
        self.config = config
        self.last_image = None
        if first_image is not None:
            assert first_time is not None
            self.init(first_image, first_time)

        self.npix = H * W

    def init(self, first_image, first_time):
        print("Initialized event camera simulator with sensor size:", first_image.shape)

        self.resolution = first_image.shape  # The resolution of the image

        # We ignore the 2D nature of the problem as it is not relevant here
        # It makes multi-core processing more straightforward
        first_image = first_image.reshape(-1)

        # Allocations
        self.last_image = first_image.copy()
        self.current_image = first_image.copy()

        self.last_time = first_time

        self.output_events = np.zeros((self.config.max_events_per_frame), dtype=EVENT_TYPE)

        self.event_count = 0
        self.spikes = np.zeros((self.npix))

    def image_callback(self, new_image, new_time):
        if self.last_image is None:
            self.init(new_image, new_time)
            return None, None

        assert new_time > 0
        assert new_image.shape == self.resolution
        new_image = new_image.reshape(-1)  # Free operation

        np.copyto(self.current_image, new_image)

        delta_time = new_time - self.last_time

        config = self.config
        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.spikes = np.zeros((self.npix))

        self.crossings = self.last_image.copy()
        self.event_count = esim(
            self.current_image.size,
            self.current_image,
            self.last_image,
            delta_time,
            self.crossings,
            self.last_time,
            self.output_events,
            self.spikes,
            config.refractory_period_ns,
            config.max_events_per_frame,
            self.W,
        )

        np.copyto(self.last_image, self.current_image)
        self.last_time = new_time

        result = self.output_events[: self.event_count]
        result.sort(order=["timestamp"], axis=0)

        return self.spikes, result
class AirSimEventGen:
    def __init__(self, W, H, save=False, debug=False):
        # self.ev_sim = EventSimulator(W, H)
        self.ev_sim = EventSimulator(W, H)
        self.H = H
        self.W = W
        self.init = True
        self.start_ts = None

        self.rgb_image_shape = [H, W, 3]
        self.debug = debug
        self.save = save

        self.event_file = open("events.pkl", "ab")
        self.event_fmt = "%1.7f", "%d", "%d", "%d"

        if debug:
            self.fig, self.ax = plt.subplots(1, 1)

    def visualize_events(self, event_img):
        event_img = self.convert_event_img_rgb(event_img)
        self.ax.cla()
        self.ax.imshow(event_img, cmap="viridis")
        plt.draw()
        plt.pause(0.01)

    #visualization
    def convert_event_img_rgb(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip(image, 0, 1) * 255
        out[:, :, 2] = np.clip(image, -1, 0) * -255

        return out

    def _stop_event_gen(self, signal, frame):
        print("/nCtrl+C received. Stopping event sim...")
        self.event_file.close()
        sys.exit(0)

start_time = time.time()
# Open the video file
# # #360H
video_path = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mkv"
dt = 2857
divide = 1
output_path = 'D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_ESIM.dat'
#360L
# video_path = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mkv"
# dt = 2857
# divide = 0.1
# output_path = 'D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_ESIM.dat'
# 60H
# video_path = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mkv"
# dt = 2857*6
# divide = 1
# output_path = 'D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_ESIM.dat'
# # T1H
# video_path = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/25-77Eevee.avi"
# dt = 21153
# divide = 1
# output_path = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_ESIM.dat"
# T1H
video_path = "D:/2024/3DGS/dataset/nerf_synthetic/mic_colmap_easy/video.mp4"
dt = 1000
divide = 1
output_path = "D:/2024/3DGS/dataset/nerf_synthetic/mic_colmap_easy/Esim.dat"
# T06H
# video_path = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/25-77Eevee.avi"
# dt = 35478
# divide = 1
# output_path = "D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_ESIM.dat"

cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the video file.")
    exit()

# Get the frame dimensions
frame_height, frame_width, _ = frame.shape
W, H = frame_width, frame_height  # Set the desired width and height of the event output
event_generator = AirSimEventGen(W, H)
# Start processing the video frames
ts = 0
ev_full = EventBuffer(1)
ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
# out = cv2.VideoWriter('R_360_H_ESIM.avi',
#                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0,
#                       (W+1, H+1)) 
while True:
    # Read a frame from the video
    ret, img = cap.read()
    if not ret:
        break

    if event_generator.init:
        event_generator.start_ts = 0
        event_generator.init = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("t",img)
    # cv2.waitKey()
    # Add small number to avoid issues with log(I)
    img = img*divide
    img = cv2.add(img, 1)
    # img = img.reshape(-1, 3)

    ts = ts+dt

    # Get the current timestamp
    current_time = ts

    # Generate events from the frame
    spikes, events = event_generator.ev_sim.image_callback(img, current_time)

    # Visualize the generated events
    event_img = np.zeros((H, W))
    if events is None:
        continue
    for event in events:
        x, y, polarity = event["x"], event["y"], event["polarity"]
        event_img[y, x] = polarity
    ev = EventBuffer(1)
    ev.add_array(np.array(events["timestamp"], dtype=np.uint64),
                np.array(events["y"], dtype=np.uint16),
                np.array(events["x"], dtype=np.uint16),
                np.array(events["polarity"], dtype=np.uint64),
                10000000)
    # ed.update(ev, dt)
    ev_full.increase_ev(ev)
# out.release()
cap.release()
ev_full.write(output_path)

# Release the video capture and close the event file
cap.release()

end_time = time.time()
total_time = end_time - start_time
print("Total time of ESIM", total_time)
