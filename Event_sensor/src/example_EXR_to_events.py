"""
Main

"""
import cv2
import sys
sys.path.append("../../src")
from event_buffer import EventBuffer
from dvs_sensor import init_bgn_hist_cpp, DvsSensor
from event_display import EventDisplay
import dsi
import numpy as np
from ExrRead import read_exr_channel,calculate_intensity_from_spetral

from event_file_io import EventsData
import numpy as np
from scipy.spatial import KDTree
import math
import time
import cv2
from event_loss import *
import open3d as o3d
from skimage.transform import resize, rescale
import random
def Various_scene_gene(video_input,dat_output):
    start_time = time.time()
    filename = video_input
    lat = 100
    jit = 10
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 1000
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time_event = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, im = cap.read()
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time_event += dt
            if time_event > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write(dat_output.format(lat, jit, ref, tau, th, th_noise))
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of ICNS", total_time)


def Rotate_360_high(C = 100):
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    start_time = time.time()
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time_event = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<20:
        filename = "D:/2023/computional imaging/my_pbrt/build/Rotate_360_pbrt001"+str(i+20)+".exr"
        im = read_exr_channel(filename,"intensity",C)
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time_event += dt
            if time_event > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of PBES", total_time)
def Rotate_360_high_ICNS():
    start_time = time.time()
    filename = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mkv"
    lat = 100
    jit = 10
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time_event = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, im = cap.read()
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time_event += dt
            if time_event > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of ICNS", total_time)

def Rotate_360_high_video(C = 100):
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<32:
        filename = "D:/2023/computional imaging/my_pbrt/build/Rotate_360_pbrt001"+str(i+48)+".exr"
        im = read_exr_channel(filename,"intensity",C)
        im = im*100
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Rotate_360_high_ICNS_video():
    filename = "D:/2023/computional imaging/my_pbrt/output/Video/0028-0060.mp4v"
    lat = 100
    jit = 10
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, im = cap.read()
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))


def Rotate_360_low():
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10000
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('R_360_L_PBES.dat_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<20:
        filename = "D:/2023/computional imaging/my_pbrt/build/Rotate_360_pbrt001"+str(i+20)+".exr"
        im = read_exr_channel(filename,"intensity",10)
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Rotate_360_low_ICNS():
    filename = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mp4v"
    lat = 100
    jit = 10000
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('R_360_L_ICNS_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, im = cap.read()
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.10
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))

def Rotate_60_high():
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 5000
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 2857*6
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<20:
        filename = "D:/2023/computional imaging/my_pbrt/build/Rotate_360_pbrt001"+str(i+20)+".exr"
        im = read_exr_channel(filename,"intensity",100)
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Rotate_60_high_ICNS():
    filename = "D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/20_eevee.mp4v"
    lat = 100
    jit = 5000
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 2857*6
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, im = cap.read()
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))

def Trans_1mps_high():
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 21153
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<23:
        filename = "D:/2023/computional imaging/my_pbrt/build/slab_1mps_pbrt001"+str(i+25)+".exr"
        im = read_exr_channel(filename,"intensity",3500)
        im  = rescale(im, (0.6666, 0.6666), anti_aliasing=True)
        #im = resize(im, (720, 1280))
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        # im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Trans_1mps_high_ICNS():
    filename = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/25-77Eevee.avi"
    lat = 100
    jit = 10
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 21153
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    i = -1
    while cap.isOpened():
        ret, im = cap.read()
        i=i+1
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))

def Trans_06mps_high():
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 35478*2
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<53:
        filename = "D:/2023/computional imaging/my_pbrt/build/slab_1mps_pbrt001"+str(i+25)+".exr"
        im = read_exr_channel(filename,"intensity",3500)
        im  = rescale(im, (0.6666, 0.6666), anti_aliasing=True)
        #im = resize(im, (720, 1280))
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        # im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+2
        if i>20*2:
            break
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time >  0.1e8:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Trans_06mps_high_ICNS():
    filename = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/25-77Eevee.avi"
    lat = 100
    jit = 10
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 35478*2
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    i = -1
    while cap.isOpened():
        ret, im = cap.read()
        i=i+1
        if i%2!=0:
            continue
        if i>20*2:
            break
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))

def Trans_1mps_low():
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 1000
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 21153*2
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<35:
        filename = "D:/2023/computional imaging/my_pbrt/build/slab_1mps_pbrt001"+str(i+25)+".exr"
        im = read_exr_channel(filename,"intensity",3500)
        im  = rescale(im, (0.6666, 0.6666), anti_aliasing=True)
        #im = resize(im, (720, 1280))
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        # im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+2
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time >  0.1e8:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_PBES.dat'.format(lat, jit, ref, tau, th, th_noise))
def Trans_1mps_low_ICNS():
    filename = "D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/25-77Eevee.avi"
    lat = 100
    jit = 1000
    ref = 100
    tau = 300
    th = 0.3
    th_noise = 0.01
    cap = cv2.VideoCapture(filename)
    dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))     
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures 
    dt = 21153*2
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
    time = 0
    out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    i = -1
    while cap.isOpened():
        ret, im = cap.read()
        i=i+1
        if i%2!=0:
            continue
        if i>35:
            break
        out.write(im)
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]*0.1
        cv2.imshow("t", im)
        cv2.waitKey(1)
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    cap.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_ICNS.dat'.format(lat, jit, ref, tau, th, th_noise))

def View_3D(point_cloud):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    # Add the point cloud to the visualizer
    vis.create_window()
    vis.add_geometry(point_cloud)
    # Get the view control object
    view_control = vis.get_view_control()

    # Set the viewpoint from the left
    view_control.set_lookat([1, 0, 0])  # Look towards positive X-axis
    view_control.set_front([-1, 0, 0])  # Set negative X-axis as the front direction

    # Run the visualizer
    vis.run()
    # Destroy the visualizer window
    vis.destroy_window()
def Compare_Real_and_PBES(Realpath, simPath,time_intervel = 100000,C=100,N=31):
    """use it to test functions
    """    
    events_data = EventsData()
    events_data_IEBCS = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events(Realpath, time_intervel)
    events_data_IEBCS.read_IEBCS_events(simPath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]
    ev_data1 = events_data_IEBCS.events[0]

    #240000-270000
    point_cloud = events_data.display_events_3D(ev_data0,11000,12000)
    View_3D(point_cloud)
    point_cloud = events_data.display_events_3D(ev_data1,11000,12000)
    View_3D(point_cloud)    
    img1 = events_data.display_events(ev_data0,11000,12000)
    cv2.imshow("real",img1)
    cv2.waitKey()
    img2 = events_data.display_events(ev_data1,11000,12000)
    cv2.imshow("sim",img2)
    cv2.waitKey()
    cv2.imwrite('D:/2023/computional imaging/my_pbrt/output/change_of_C/2Dshow_{}_{}.jpg'.format(C,N), img2)


    #chamfer
    # start_time = time.time()
    # loss_same = chamfer_distance_loss(ev_data0, ev_data0)
    # print(loss_same)
    # loss_different = chamfer_distance_loss(ev_data0, ev_data1)
    # print(loss_different)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time of chamfer method", total_time)
    # # gausian
    # start_time = time.time()
    # loss_same = gaussian_distance_loss(ev_data0, ev_data0)
    # print(loss_same)
    # loss_different = gaussian_distance_loss(ev_data0, ev_data1)
    # print(loss_different)
    
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time of gausian method", total_time)

    # kernel
    # start_time = time.time()
    # loss_different = kernel_method_spike_cubes_loss(ev_data0, ev_data1,events_data.width,events_data.height,32,32,2500,5,5,1000)
    # print(loss_different)
    
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time of kernel method", total_time)

def Compare_Real_and_V2E(Realpath, simPath,time_intervel = 100000):
    """use it to test functions
    """    
    events_data = EventsData()
    events_data_V2E = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events(Realpath, time_intervel)
    events_data_V2E.read_V2E_events(simPath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]
    ev_data1 = events_data_V2E.events[0]


    point_cloud = events_data.display_events_3D(ev_data0,240000,270000)
    View_3D(point_cloud)
    point_cloud = events_data.display_events_3D(ev_data1,240000,270000)
    View_3D(point_cloud)
    img1 = events_data.display_events(ev_data0,240000,270000)
    cv2.imshow("real",img1)
    cv2.waitKey()
    img2 = events_data.display_events(ev_data1,240000,270000)
    # cv2.imshow("sim",img2)
    # cv2.waitKey()


    #chamfer
    start_time = time.time()
    loss_same = chamfer_distance_loss(ev_data0, ev_data0)
    print(loss_same)
    loss_different = chamfer_distance_loss(ev_data0, ev_data1)
    print(loss_different)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of chamfer method", total_time)
    # gausian
    start_time = time.time()
    loss_same = gaussian_distance_loss(ev_data0, ev_data0)
    print(loss_same)
    loss_different = gaussian_distance_loss(ev_data0, ev_data1)
    print(loss_different)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of gausian method", total_time)

    # kernel
    # start_time = time.time()
    # loss_different = kernel_method_spike_cubes_loss(ev_data0, ev_data1,events_data.width,events_data.height)
    # print(loss_different)
    
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time of kernel method", total_time)

def Rotate_360_high_Ctest(C = 100,N=31):
    #TODO,add formal ICNS here, deal v2e in other place
    # filename = "C:/Users//Documents/blender/PBES_small/rotate_360/0000-0060.mp4v"
    #lower lat will get more active pixel(remain event)
    lat = 10
    # higher jit will get blurer pixel and more active pixel , very obvious
    jit = 10
    #lower ref, higher active
    ref = 100
    #it's the trans func, higher tau , lowwer the event gen
    #only 2 chamfer in 300, best!
    #use loss to calibrate , not img, img is fucking
    tau = 300
    #just threshould, lower it can let it more realistic(not straight event)
    #very obvious!
    th = 0.3
    #higher it, more active is
    th_noise = 0.01
    # cap = cv2.VideoCapture(filename)
    #only first is hw, other are wh

    #rotate_360_high, 100-10=100-3-0.3-0.01
    width = 1280
    height = 720
    dsi.initSimu(int(height), int(width))
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    #the noise can also be better!
    init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy")
    isInit = False
    # dt = 1000  # FPS must be 1 kHz,this means fps = dt, and it shows that it can capture high fpx, like bird wings.
            #however, I do not need so high frequency, if I can't output somany pitures
    # if dt is too high, result is wrong? 
    # if bigger than 2857*2, result is wrong, I don't know why
    # if try to look better, may be small dt in 360...
    #if you change dt, t total is longer, that's right!
    dt = 2857
    ev_full = EventBuffer(1)
    ed = EventDisplay("Events", width, height, dt)
    time = 0
    out = cv2.VideoWriter('rotate_360_highlight_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(width), int(height)))
    i = 0
    while i<10:
        filename = "D:/2023/computional imaging/my_pbrt/build/Rotate_360_pbrt001"+str(i+20)+".exr"
        if N==32:
            im = read_exr_channel(filename,"intensity",C)
        else:
            im = calculate_intensity_from_spetral(filename,N,C)
        cv2.imshow("t", im)
        cv2.waitKey(1)
        im = im*750
        #value must near 127,otherwise it will get wrong answner
        #becaose backend suppose it as 0-255, you should not use 0-1
        out.write(im)
        i=i+1
        if im is None:
            break
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        if not isInit:
            dsi.initImg(im)
            isInit = True
        else:
            buf = dsi.updateImg(im, dt)
            ev = EventBuffer(1)
            ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                            np.array(buf["x"], dtype=np.uint16),
                            np.array(buf["y"], dtype=np.uint16),
                            np.array(buf["p"], dtype=np.uint64),
                            1000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
            if time > 0.1e7:
                break
    out.release()
    ev_full.write('D:/2023/computional imaging/my_pbrt/output/change_of_C/R_360_H_PBES_CTEST_{}_{}.dat'.format(C,N))
def Get_Result_Of_DIfferent_C_N(C = 100,N=31):
    print(N)
    # start_time = time.time()
    Rotate_360_high_Ctest(C,N)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time of N hyper", total_time)
    #Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5","D:/2023/computional imaging/my_pbrt/output/change_of_C/R_360_H_PBES_CTEST_{}_{}.dat".format(C,N),100000,C,N)

def Compare_Real_and_PBES_biast(Realpath,time_intervel = 100000,bias = 1000,C=100,N=31):
    events_data = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events(Realpath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]


    # Calculate the increasing bias for each event in ev_data0
    num_events = ev_data0.size
    t_bias = np.linspace(0, bias, num_events)
    # Shift ev_data0 by bias on the t axis to obtain ev_data1
    ev_data1 = ev_data0.copy()
    ev_data1['t'] += t_bias.astype(np.int64)

    #View
    point_cloud = events_data.display_events_3D(ev_data0,0,100000)
    View_3D(point_cloud)
    point_cloud = events_data.display_events_3D(ev_data1,0,100000)
    View_3D(point_cloud)    
    #chamfer

    print("Bias of t is", bias)
    start_time = time.time()
    loss_different = chamfer_distance_loss(ev_data0, ev_data1)
    print(loss_different)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of chamfer method", total_time)
    # gausian
    start_time = time.time()
    loss_different = gaussian_distance_loss(ev_data0, ev_data1)
    print(loss_different)

def Compare_Real_and_PBES_noise(Realpath,time_intervel = 100000,noise_number = 1000,C=100,N=31):
    events_data = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events(Realpath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]



    # Generate noise points on x, y, and t axes
    num_events = ev_data0.size
    ev_data1 = np.empty(num_events + noise_number, dtype=ev_data0.dtype)
    ev_data1[:num_events] = ev_data0

    step = num_events // noise_number

    min_x = 0
    max_x = events_data.width - 1
    min_y = 0
    max_y = events_data.height - 1
    for point in range(noise_number):
        ev_data1[num_events + point]['x'] = random.randint(min_x, max_x)
        ev_data1[num_events + point]['y'] = random.randint(min_y, max_y)
        ev_data1[num_events + point]['t'] = ev_data0[point * step]['t']
        ev_data1[num_events + point]['p'] = point % 2  # Alternate between p=0 and p=1

    # Combine ev_data0 and noise_points to create ev_data1
    point_cloud = events_data.display_events_3D(ev_data0,0,5000)
    View_3D(point_cloud)
    point_cloud = events_data.display_events_3D(ev_data1,0,5000)
    View_3D(point_cloud)    
    #chamfer

    print("Noise point number is", noise_number)
    start_time = time.time()
    loss_different = chamfer_distance_loss(ev_data0, ev_data1)
    print(loss_different)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of chamfer method", total_time)
    # gausian
    start_time = time.time()
    loss_different = gaussian_distance_loss(ev_data0, ev_data1)
    print(loss_different)

def Generate_display_video(Realpath,VideoName,time_intervel = 100000):
    events_data = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_IEBCS_events(Realpath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]
    events_data.generate_video(ev_data0,2857,events_data.global_max_t-2857*3,2857/1,VideoName,5)
def Generate_display_video_v2e(Realpath,VideoName,time_intervel = 100000):
    events_data = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_V2E_events(Realpath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]
    events_data.generate_video(ev_data0,2857,events_data.global_max_t-2857*3,2857/1,VideoName,5)

def Generate_display_video_real(Realpath,VideoName,time_intervel = 100000):
    events_data = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events(Realpath, time_intervel)
    #3D output is the best way to calibrate
    ev_data0 = events_data.events[0]
    events_data.generate_video(ev_data0,2857*2.5,events_data.global_max_t-2857*0.5,2857/1.02,VideoName,5)

def Generate_display_video_final(VideoFolder):
    events_data = EventsData()
    events_data.generate_display_video(VideoFolder,20)

# Various_scene_gene("D:/2023/computional imaging/my_pbrt/output/Various_Scene/dragon_0000-0078.mkv","D:/2023/computional imaging/my_pbrt/output/Various_Scene/dragon_0000-0078.dat")
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Various_Scene/dragon_0000-0078.dat","D:/2023/computional imaging/my_pbrt/output/Various_Scene/dragon_0000-0078.mp4v",100000)
# Various_scene_gene("D:/2023/computional imaging/my_pbrt/output/Various_Scene/city2_0001-0060.mkv","D:/2023/computional imaging/my_pbrt/output/Various_Scene/city2_0001-0060.dat")
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Various_Scene/city2_0001-0060.dat","D:/2023/computional imaging/my_pbrt/output/Various_Scene/city2_0001-0060.mp4v",100000)
# Various_scene_gene("D:/2023/computional imaging/my_pbrt/output/Various_Scene/city0001-0060.mkv","D:/2023/computional imaging/my_pbrt/output/Various_Scene/city0001-0060.dat")
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Various_Scene/city0001-0060.dat","D:/2023/computional imaging/my_pbrt/output/Various_Scene/city0001-0060.mp4v",100000)
# Various_scene_gene("D:/2023/computional imaging/my_pbrt/output/Various_Scene/tree.mkv","D:/2023/computional imaging/my_pbrt/output/Various_Scene/tree.dat")
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Various_Scene/tree.dat","D:/2023/computional imaging/my_pbrt/output/Various_Scene/tree.mp4v",100000)


#Rotate_360_high_video()
#Rotate_360_high_ICNS_video()

# Generate_display_video_real("D:/2023/computional imaging/my_pbrt/output/Video/High_360_360.hdf5","D:/2023/computional imaging/my_pbrt/output/Video/High_360_360deg.mp4v",100000)
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_PBES.dat","D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_PBES.mp4v",100000)
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_ICNS.dat","D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_ICNS.mp4v",100000)
# Generate_display_video("D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_ESIM.dat","D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_ESIM.mp4v",100000)
# Generate_display_video_v2e("D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_V2E.txt","D:/2023/computional imaging/my_pbrt/output/Video/R_360_H_V2E.mp4v",100000)
Generate_display_video_final("D:/2023/computional imaging/my_pbrt/output/Video/")

# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,0)
# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,10)
# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,100)
# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,1000)
# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,10000)
# Compare_Real_and_PBES_biast("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,100000)

# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,0)
# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,10)
# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,100)
# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,1000)
# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,10000)
# Compare_Real_and_PBES_noise("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",10000,100000)

# Get_Result_Of_DIfferent_C_N(8,31)
# Get_Result_Of_DIfferent_C_N(10,31)
# Get_Result_Of_DIfferent_C_N(30,31)
# Get_Result_Of_DIfferent_C_N(60,31) 
# Get_Result_Of_DIfferent_C_N(80,31) 
# Get_Result_Of_DIfferent_C_N(100,31) 
# Get_Result_Of_DIfferent_C_N(120,31) 
# Get_Result_Of_DIfferent_C_N(140,31)
# Get_Result_Of_DIfferent_C_N(160,31)
# Get_Result_Of_DIfferent_C_N(180,31)   
# Get_Result_Of_DIfferent_C_N(220,31)
# Get_Result_Of_DIfferent_C_N(240,31)
# Get_Result_Of_DIfferent_C_N(260,31)
# Get_Result_Of_DIfferent_C_N(300,31)
# Get_Result_Of_DIfferent_C_N(480,31)
# Get_Result_Of_DIfferent_C_N(1000,31)
# Get_Result_Of_DIfferent_C_N(1920,31)
# Get_Result_Of_DIfferent_C_N(100000,31)

#N
# Get_Result_Of_DIfferent_C_N(100,31)
# Get_Result_Of_DIfferent_C_N(100,16)
# Get_Result_Of_DIfferent_C_N(100,11)
# Get_Result_Of_DIfferent_C_N(100,7)
# Get_Result_Of_DIfferent_C_N(100,6) 
#8，30，120，480，1920
# Rotate_360_high()
# Rotate_360_high_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5","D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_PBES.dat",100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_ICNS.dat',100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_ESIM.dat',100000)
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/High_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_high/R_360_H_V2E.txt')

# Rotate_360_low()
# Rotate_360_low_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/LOW_360_120deg.hdf5","D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_PBES.dat",100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/LOW_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_ICNS.dat',100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/LOW_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_ESIM.dat')
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/LOW_360_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_360_low/R_360_L_V2E.txt')


# Rotate_60_high()
# Rotate_60_high_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/High_60_120deg.hdf5","D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_PBES.dat",100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/High_60_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_ICNS.dat',100000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/High_60_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_ESIM.dat',)
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/High_60_120deg.hdf5",'D:/2023/computional imaging/my_pbrt/output/Rotate_60_high/R_60_H_V2E.txt',)

# Trans_1mps_high()
# Trans_1mps_high_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/trans_1mps_H.hdf5","D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_PBES.dat",200000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/trans_1mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_ICNS.dat',200000)
#Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/trans_1mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_ESIM.dat',300000)
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/trans_1mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_high/T_1_H_v2e.txt',200000)


# Trans_06mps_high()
# Trans_06mps_high_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/trans_06mps_H.hdf5","D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_PBES.dat",300000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/trans_06mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_ICNS.dat',300000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/trans_06mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_ESIM.dat',300000)
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/trans_06mps_H.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_06mps_high/T_06_H_v2e.txt',300000)

# Trans_1mps_low()
# Trans_1mps_low_ICNS()
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/trans_1mps_L.hdf5","D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_PBES.dat",200000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/trans_1mps_L.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_ICNS.dat',200000)
# Compare_Real_and_PBES("D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/trans_1mps_L.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_ESIM.dat',200000)
# Compare_Real_and_V2E("D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/trans_1mps_L.hdf5",'D:/2023/computional imaging/my_pbrt/output/trans_1mps_low/T_1_L_v2e.txt',200000)