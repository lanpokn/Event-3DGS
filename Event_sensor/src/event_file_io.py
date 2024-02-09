#P 1 positive, 0 or -1 negative, it may different in different simulator
#thus it's important to use 0.5 or =1 to deal with 
#white positive
#in real camera: 1280 in first dimen, 720 in second dim, height is 720(second dim)
#in sim camera also 1280 first, 720 second
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
import os
import cv2
import numpy as np
import open3d as o3d
import torch
#We don’t provide any encoder for EVT3 format as its compressed nature makes it difficult to be re-created from an uncompressed CSV file
# class EventData:
    
# class EventDataset:
    

import argparse
from typing import List

import struct

from metavision_core.event_io import DatWriter

import h5py

def load_h5_event(input_path):
    with h5py.File(input_path, 'r') as f:
        ts = f['t'][:]
        x = f['x'][:]
        y = f['y'][:]
        p = f['p'][:]
    return ts, x, y, p
def write_meta_dat(filename, ts, x, y, pol, width=None, height=None):
    if width is None:
        width = np.max(x) + 1
    if height is None:
        height = np.max(y) + 1
    # Combine ts, x, y, and pol into a structured NumPy array
    events_data = np.array(list(zip(ts, x, y, pol)), dtype=[('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])

    # Create an instance of H5EventsWriter
    Writer =  DatWriter(filename, height=height, width=width)
    # Write events to the HDF5 file
    Writer.write(events_data)

def load_dat_event(filename, start=0, stop=-1, display=False):
    """ Load .dat event.
        Warning: only tested on the VGA sensor on V2 Prophesee event
        Args:
            filename: Path of the .dat file
            start: starting timestamp (us)
            stop: if different than -1, last timestamp
            display: display file infos
        Returns:
             ts, x, y, pol numpy arrays of timestamps, position and polarities
     """
    f = open(filename, 'rb')
    if f == -1:
        print("The file does not exist")
        return
    else:
        if display: print("Load DAT Events: " + filename)
    l = f.readline()
    all_lines = l
    while l[0] == 37:
        p = f.tell()
        if display: print(l)
        l = f.readline()
        all_lines = all_lines + l
    f.close()
    all_lines = str(all_lines)
    f = open(filename, 'rb')
    f.seek(p, 0)
    evType = np.uint8(f.read(1)[0])
    evSize = np.uint8(f.read(1)[0])
    p = f.tell()
    l_last = f.tell()
    if start > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < start:
            p = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])

    if stop > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < stop:
            l_last = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
    else:
        l_last = f.seek(0, 2)

    num_b = (l_last - p) // evSize * 2
    f.close()
    data = np.fromfile(filename, dtype=np.uint32, count=num_b, offset=p)
    ts = data[::2]
    v = 0
    ind = all_lines.find("Version")
    if ind > 0:
        v = int(all_lines[ind+8])
    if v >= 2:
        x_mask = np.uint32(0x00007FF)
        y_mask = np.uint32(0x0FFFC000)
        pol_mask = np.uint32(0x10000000)
        x_shift = 0
        y_shift = 14
        pol_shift = 28
    else:
        x_mask = np.uint32(0x00001FF)
        y_mask = np.uint32(0x0001FE00)
        pol_mask = np.uint32(0x00020000)
        x_shift = 0
        y_shift = 9
        pol_shift = 17
    x = data[1::2] & x_mask
    x = x >> x_shift
    y = data[1::2] & y_mask
    y = y >> y_shift
    pol = data[1::2] & pol_mask
    pol = pol >> pol_shift
    if len(ts) > 0:
        if display: 
            print("First Event: ", ts[0], " us")
            print("Last Event: ", ts[-1], " us")
            print("Number of Events: ", ts.shape[0])
    return ts, x, y, pol


# evs is a single unit, which can be the input of loss, display and so on
class EventsData:
    def __init__(self):
        #x,y,t,p,type is ndarray
        self.events = []
        self.global_counter = 0
        self.global_max_t = -1
        self.global_min_t = -1
        self.delta_t = 0
        self.width = 0
        self.height = 0
    #delta_t is used to split original data
    def read_real_events(self, input_path: str,delta_t_input:int):
        """Process events and update EventsData object"""
        self.delta_t = delta_t_input
        mv_iterator = EventsIterator(input_path=input_path, delta_t = delta_t_input)
        self.height, self.width = mv_iterator.get_size()  # Camera Geometry
        for evs in mv_iterator:
            if evs.size == 0:
                continue
            if self.global_min_t ==-1:
                self.global_min_t = evs['t'][0]
            self.global_max_t = evs['t'][-1]
            counter = evs.size
            
            self.events.append(evs)
            self.global_counter += counter
        evs['t'] = evs['t']-self.global_min_t
        self.global_max_t = self.global_max_t-self.global_min_t
        self.global_min_t = 0
    def read_eds_events(self, input_path: str,delta_t_input:int):
        # input path should be xxx .dat
        """Process events and update EventsData object"""
        ts, x, y, p = load_h5_event(input_path)
        self.delta_t = delta_t_input
        #x max is 
        self.height = np.max(y) + 1  # Set self.height as the maximum value of x plus 1
        self.width = np.max(x) + 1   # Set self.width as the maximum value of y plus 1
        
        #0 start, or it's too hard to tell range
        # ts = ts - ts[0]

        start_time = ts[0]
        end_time = ts[-1]
        time_duration = end_time - start_time
        #0 start, or it's too hard to tell range
        
        num_buffers = time_duration // delta_t_input

        for i in range(num_buffers):
            start_idx = np.searchsorted(ts, start_time + i * delta_t_input)
            end_idx = np.searchsorted(ts, start_time + (i + 1) * delta_t_input, side='right')
            #evs = np.zeros(end_idx - start_idx, dtype=[('x', 'int32'), ('y', 'int32'), ('t', 'int32'), ('p', 'int32')])
            evs = np.zeros(end_idx - start_idx, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))            
            evs['x'] = x[start_idx:end_idx]
            evs['y'] = y[start_idx:end_idx]
            evs['t'] = ts[start_idx:end_idx]
            evs['p'] = p[start_idx:end_idx]
            self.events.append(evs)
            self.global_counter += evs.size

        remaining_events = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
        if remaining_events.size > 0:
            #evs = np.zeros(remaining_events.size, dtype=[('x', 'int32'), ('y', 'int32'), ('t', 'int32'), ('p', 'int32')])
            evs = np.zeros(remaining_events.size, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))            
            evs['x'] = x[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['y'] = y[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['t'] = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['p'] = p[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            self.events.append(evs)
            self.global_counter += evs.size

        if self.global_min_t == -1:
            self.global_min_t = ts[0]
        self.global_max_t = ts[-1]
        evs['t'] = evs['t']  -self.global_min_t
        self.global_max_t = self.global_max_t-self.global_min_t
        self.global_min_t = 0
    def read_IEBCS_events(self, input_path: str,delta_t_input:int):
        # input path should be xxx .dat
        """Process events and update EventsData object"""
        ts, x, y, p = load_dat_event(input_path)
        self.delta_t = delta_t_input
        #x max is 
        self.height = np.max(y) + 1  # Set self.height as the maximum value of x plus 1
        self.width = np.max(x) + 1   # Set self.width as the maximum value of y plus 1

        start_time = ts[0]
        end_time = ts[-1]
        time_duration = end_time - start_time

        num_buffers = time_duration // delta_t_input

        for i in range(num_buffers):
            start_idx = np.searchsorted(ts, start_time + i * delta_t_input)
            end_idx = np.searchsorted(ts, start_time + (i + 1) * delta_t_input, side='right')
            #evs = np.zeros(end_idx - start_idx, dtype=[('x', 'int32'), ('y', 'int32'), ('t', 'int32'), ('p', 'int32')])
            evs = np.zeros(end_idx - start_idx, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))            
            evs['x'] = x[start_idx:end_idx]
            evs['y'] = y[start_idx:end_idx]
            evs['t'] = ts[start_idx:end_idx]
            evs['p'] = p[start_idx:end_idx]
            self.events.append(evs)
            self.global_counter += evs.size

        remaining_events = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
        if remaining_events.size > 0:
            #evs = np.zeros(remaining_events.size, dtype=[('x', 'int32'), ('y', 'int32'), ('t', 'int32'), ('p', 'int32')])
            evs = np.zeros(remaining_events.size, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))            
            evs['x'] = x[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['y'] = y[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['t'] = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['p'] = p[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            self.events.append(evs)
            self.global_counter += evs.size

        if self.global_min_t == -1:
            self.global_min_t = ts[0]
        self.global_max_t = ts[-1]
        evs['t'] = evs['t']  -self.global_min_t
        self.global_max_t = self.global_max_t-self.global_min_t
        self.global_min_t = 0

    def read_V2E_events(self, input_path: str, delta_t_input: int):
        # max size: 5000000, if you need more, delete that code
        with open(input_path, 'r') as file:
            lines = file.readlines()

        ts, x, y, p = [], [], [], []
        i = 0
        for line in lines:
            if line.startswith('#') or line.startswith('('):
                continue
            values = line.strip().split()
            ts.append(int(float(values[0]) * 1e6))
            x.append(int(values[1]))
            y.append(int(values[2]))
            p.append(int(values[3]))
            i=i+1
            if i>5000000:
                break

        self.delta_t = delta_t_input
        self.height = np.max(y) + 1
        self.width = np.max(x) + 1
        start_time = ts[0]
        end_time = ts[-1]
        time_duration = end_time - start_time
        num_buffers = time_duration // delta_t_input

        for i in range(num_buffers):
            start_idx = np.searchsorted(ts, start_time + i * delta_t_input)
            end_idx = np.searchsorted(ts, start_time + (i + 1) * delta_t_input, side='right')
            evs = np.zeros(end_idx - start_idx, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))
            evs['x'] = x[start_idx:end_idx]
            evs['y'] = y[start_idx:end_idx]
            evs['t'] = ts[start_idx:end_idx]
            evs['p'] = p[start_idx:end_idx]
            self.events.append(evs)
            self.global_counter += evs.size

        remaining_events = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
        if len(remaining_events) > 0:
            evs = np.zeros(len(remaining_events), dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))
            evs['x'] = x[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['y'] = y[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['t'] = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['p'] = p[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            self.events.append(evs)
            self.global_counter += len(evs)

        if self.global_min_t == -1:
            self.global_min_t = ts[0]
        self.global_max_t = ts[-1]
        evs['t'] = evs['t'] - self.global_min_t
        self.global_max_t = self.global_max_t - self.global_min_t
        self.global_min_t = 0
    def read_Volt_events(self, input_path: str, delta_t_input: int):
        # max size: 5000000, if you need more, delete that code
        with open(input_path, 'r') as file:
            lines = file.readlines()

        ts, x, y, p = [], [], [], []
        i = 0
        for line in lines:
            if line.startswith('#') or line.startswith('('):
                continue
            values = line.strip().split()
            ts.append(int(float(values[0])))
            x.append(int(values[1]))
            y.append(int(values[2]))
            p.append(int(values[3]))
            i=i+1
            if i>5000000:
                break

        self.delta_t = delta_t_input
        self.height = np.max(y) + 1
        self.width = np.max(x) + 1
        start_time = ts[0]
        end_time = ts[-1]
        time_duration = end_time - start_time
        num_buffers = time_duration // delta_t_input

        for i in range(num_buffers):
            start_idx = np.searchsorted(ts, start_time + i * delta_t_input)
            end_idx = np.searchsorted(ts, start_time + (i + 1) * delta_t_input, side='right')
            evs = np.zeros(end_idx - start_idx, dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))
            evs['x'] = x[start_idx:end_idx]
            evs['y'] = y[start_idx:end_idx]
            evs['t'] = ts[start_idx:end_idx]
            evs['p'] = p[start_idx:end_idx]
            self.events.append(evs)
            self.global_counter += evs.size

        remaining_events = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
        if len(remaining_events) > 0:
            evs = np.zeros(len(remaining_events), dtype=np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}))
            evs['x'] = x[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['y'] = y[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['t'] = ts[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            evs['p'] = p[np.searchsorted(ts, start_time + num_buffers * delta_t_input):]
            self.events.append(evs)
            self.global_counter += len(evs)

        if self.global_min_t == -1:
            self.global_min_t = ts[0]
        self.global_max_t = ts[-1]
        evs['t'] = evs['t'] - self.global_min_t
        self.global_max_t = self.global_max_t - self.global_min_t
        self.global_min_t = 0

    def display_events_metavision(self,evs,accumulation_time_us):
        # Window - Graphical User Interface
        # evs must one of the EventsData, or h and w are incorrect
        with Window(title="Metavision SDK Get Started", width=self.width, height=self.height, mode=BaseWindow.RenderMode.BGR) as window:
            def keyboard_cb(key, scancode, action, mods):
                if action != UIAction.RELEASE:
                    return
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    window.set_close_flag()

            window.set_keyboard_callback(keyboard_cb)
            event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=self.width, sensor_height=self.height,
                                                    accumulation_time_us=accumulation_time_us)

            def on_cd_frame_cb(ts, cd_frame):
                window.show(cd_frame)

            event_frame_gen.set_output_callback(on_cd_frame_cb)
            
            EventLoop.poll_and_dispatch()
            window_cv = cv2.namedWindow("click to show", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("click to show",self.width,self.height)
            cv2.imshow("click to show", 255)

            # Wait for a key press
            cv2.waitKey(0)

            # Close the window
            cv2.destroyAllWindows()
            event_frame_gen.process_events(evs)
            
            # if window.should_close():
            #     break
    def display_events(self,events,t_begin,t_end):
        width = self.width
        height = self.height
        img = 255 * np.ones((height, width, 3), dtype=np.uint8)
        
        events_filtered = events[(events['t'] >= t_begin) & (events['t'] <= t_end)]  # Filter events based on time
        
        if events_filtered.size:
            assert events_filtered['x'].max() < width, "out of bound events: x = {}, w = {}".format(events_filtered['x'].max(), width)
            assert events_filtered['y'].max() < height, "out of bound events: y = {}, h = {}".format(events_filtered['y'].max(), height)
            
            ON_index = np.where(events_filtered['p'] == 1)
            img[events_filtered['y'][ON_index], events_filtered['x'][ON_index], :] = [30, 30, 220] * events_filtered['p'][ON_index][:, None]  # red [0, 0, 255]#bgr
            
            OFF_index = np.where(events_filtered['p'] == 0)
            img[events_filtered['y'][OFF_index], events_filtered['x'][OFF_index], :] = [200, 30, 30] * (events_filtered['p'][OFF_index] + 1)[:, None]  # green [0, 255, 0], blue [255, 0, 0]
        
        return img
    def display_events_torch(self, events, t_begin, t_end):
        width = self.width
        height = self.height
        img = 255 * torch.ones((3,height, width), dtype=torch.float32)

        events_filtered = events[(events['t'] >= t_begin) & (events['t'] <= t_end)]  # Filter events based on time
        
        if events_filtered.size:
            assert events_filtered['x'].max() < width, "out of bound events: x = {}, w = {}".format(events_filtered['x'].max(), width)
            assert events_filtered['y'].max() < height, "out of bound events: y = {}, h = {}".format(events_filtered['y'].max(), height)
            
            ON_index = np.where(events_filtered['p'] == 1)
            img[:,events_filtered['y'][ON_index], events_filtered['x'][ON_index]] = torch.tensor([30, 30, 220], dtype=torch.float32) 
            
            OFF_index = np.where(events_filtered['p'] == 0)
            img[:,events_filtered['y'][OFF_index], events_filtered['x'][OFF_index]] = torch.tensor([30, 30, 220], dtype=torch.float32) 
        return img
    def display_events_accumu(self, events, t_begin, t_end, width=1280, height=720):
        width = self.width
        height = self.height
        img = np.zeros((height, width, 3), dtype=np.uint8)

        events_filtered = events[(events['t'] >= t_begin) & (events['t'] <= t_end)]  # Filter events based on time

        # 将 events_filtered['x'], events_filtered['y'] 转换为整数类型
        x_values = events_filtered['x'].astype(int)
        y_values = events_filtered['y'].astype(int)

        # 创建一个零矩阵，用于存储 on 和 off 的数量
        on_count = np.zeros((height, width), dtype=np.uint8)
        off_count = np.zeros((height, width), dtype=np.uint8)

        # 计算 on 和 off 的数量
        np.add.at(on_count, (y_values, x_values), (events_filtered['p'] == 1))
        np.add.at(off_count, (y_values, x_values), (events_filtered['p'] == 0))

        img[:, :, 2] = on_count*10  # Store on counts in the third channel
        img[:, :, 0] = off_count*10  # Store off counts in the first channel

        return img
    def generate_video(self, events, t_begin, t_end, dt=2857*2,video_name = "default",cycles = 1,width=1280, height=720):
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Define codec for video writer
        video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # Create video writer object
        loop = cycles
        while loop > 0:
            t_current = t_begin
            while t_current <= t_end:
                img = self.display_events(events, t_current, t_current + dt, width, height)  # Generate frame using display_events function
                video.write(img)  # Write frame to video
                t_current += dt
            loop = loop-1

        video.release()  # Release video writer object
    def generate_display_video(self,videoPath, fps):
        cap_rgb = cv2.VideoCapture(videoPath+"0000-0120.mkv")
        cap_real = cv2.VideoCapture(videoPath+"High_360_360deg.mp4v")
        cap_PECS = cv2.VideoCapture(videoPath+"R_360_H_PBES.mp4v")
        cap_ICNS = cv2.VideoCapture(videoPath+"R_360_H_ICNS.mp4v")
        cap_V2E = cv2.VideoCapture(videoPath+"R_360_H_V2E.mp4v")
        cap_ESIM = cv2.VideoCapture(videoPath+"R_360_H_ESIM.mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        l2r = 640
        h2l = 360
        videoWriter = cv2.VideoWriter(videoPath+"display.avi", fourcc, fps, (l2r*2, h2l*3), True)
        while cap_rgb.isOpened() and cap_real.isOpened() and cap_PECS.isOpened():
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_real, frame_real = cap_real.read()
            ret_PECS, frame_PECS = cap_PECS.read()
            ret_ICNS, frame_ICNS = cap_ICNS.read()
            ret_V2E, frame_V2E = cap_V2E.read()
            ret_ESIM, frame_ESIM = cap_ESIM.read()
            # Resize frames to l2rx360
            if ret_rgb and ret_real and ret_PECS:
                frame_rgb = cv2.copyMakeBorder(frame_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_real = cv2.copyMakeBorder(frame_real, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_PECS = cv2.copyMakeBorder(frame_PECS, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_ICNS = cv2.copyMakeBorder(frame_ICNS, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_V2E = cv2.copyMakeBorder(frame_V2E, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_ESIM = cv2.copyMakeBorder(frame_ESIM, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
                frame_rgb = cv2.resize(frame_rgb, (l2r, h2l))
                frame_real = cv2.resize(frame_real, (l2r, h2l))
                frame_PECS = cv2.resize(frame_PECS, (l2r, h2l))
                frame_ICNS = cv2.resize(frame_ICNS, (l2r, h2l))
                frame_V2E = cv2.resize(frame_V2E, (l2r, h2l))
                frame_ESIM = cv2.resize(frame_ESIM, (l2r, h2l))

                first_column_image = np.hstack((frame_rgb, frame_real))
                second_column_image = np.hstack((frame_PECS, frame_ICNS))
                third_column_image = np.hstack((frame_V2E, frame_ESIM))
                all_frame = np.vstack((first_column_image,second_column_image,third_column_image))
                cv2.putText(all_frame, 'RGB image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                cv2.putText(all_frame, 'Real Event in EVK4', (l2r+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                cv2.putText(all_frame, 'our PECS', (10, 30+h2l), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                cv2.putText(all_frame, 'ICNS', (l2r+10, 30+h2l), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                cv2.putText(all_frame, 'V2E', (10, 30+h2l*2), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                cv2.putText(all_frame, 'ESIM', (l2r+10, 30+h2l*2), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 0, 255), 2)
                # print(all_frame[0])
                videoWriter.write(all_frame)
            else:
                break
        cap_rgb.release()
        cap_ESIM.release()
        cap_PECS.release()
        cap_ICNS.release()
        cap_V2E.release()
        cap_real.release()
        videoWriter.release()
    def display_events_3D(self,events,t_begin,t_end,width=1280, height=720):
        # Create a point cloud object
        point_cloud = o3d.geometry.PointCloud()

        # Filter events based on time
        events_filtered = events[(events['t'] >= t_begin) & (events['t'] <= t_end)]

        if events_filtered.size:
            # Set the points' positions and colors
            t = events_filtered['t']
            positions = np.column_stack((events_filtered['x'], events_filtered['y'], t))
            colors = np.zeros(positions.shape)
            colors[:, 0] = 30  # Set red color for ON events
            colors[:, 2] = 220  # Set blue color for OFF events

            # Set the points' colors based on p values
            ON_index = np.where(events_filtered['p'] == 1)
            colors[ON_index, :] = [30, 30, 220]  # Set red color for ON events

            OFF_index = np.where(events_filtered['p'] == 0)
            colors[OFF_index, :] = [200, 30, 30]  # Set green color for OFF events

            # Assign positions and colors to the point cloud
            point_cloud.points = o3d.utility.Vector3dVector(positions)
            point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud])
        return point_cloud





def main():
    events_data = EventsData()
    # events_data.read_real_events("C:/Users/admin/Documents/metavision/recordings/output1.hdf5",1000)
    events_data.read_IEBCS_events("D:/2023/计算成像与仿真/my_pbrt/IEBCS-main/ev_100_10_100_300_0.3_0.01.dat",10000)
    for ev_data in events_data.events:
        print("----- New event buffer! -----")
        counter = ev_data.size
        min_t = ev_data['t'][0]
        max_t = ev_data['t'][-1]
        print(f"There were {counter} events in this event buffer.")
        print(f"There were {events_data.global_counter} total events up to now.")
        print(f"The current event buffer included events from {min_t} to {max_t} microseconds.")
        print("----- End of the event buffer! -----")

    duration_seconds = events_data.global_max_t / 1.0e6
    print(f"There were {events_data.global_counter} events in total.")
    print(f"The total duration was {duration_seconds:.2f} seconds.")
    if duration_seconds >= 1:
        print(f"There were {events_data.global_counter / duration_seconds:.2f} events per second on average.")


if __name__ == "__main__":
    main()