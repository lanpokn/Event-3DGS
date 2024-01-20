import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import sys
sys.path.append("./Event_sensor")
sys.path.append("./Event_sensor/src")
from src.event_buffer import EventBuffer
from src.dvs_sensor import init_bgn_hist_cpp, DvsSensor
from src.event_display import EventDisplay
import dsi
import numpy as np
from src.event_file_io import EventsData
import os
def rotation_matrix_to_quaternion(rotation_matrix):
    # 使用scipy的Rotation类来将旋转矩阵转换为四元数
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion

def quaternion_to_rotation_matrix(quaternion):
    # 使用scipy的Rotation类来将四元数转换为旋转矩阵
    r = Rotation.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def simulate_event_camera(images,ev_full,dt=2857,lat=100, jit=10, ref=100, tau=300, th=0.3, th_noise=0.01):
    dsi.initSimu(images[0].shape[0], images[0].shape[1])
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)

    isInit = False
    time = 0

    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        # cv2.imshow("t", im)
        # cv2.waitKey(1)
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
            ev_full.increase_ev(ev)
            time += dt
    return ev_full

def save_event_result(ev_full,event_path):
    file_path = os.path.join(event_path,"raw.dat")
    ev_full.write(file_path)
    return file_path
    
def generate_images(event_path,dt, total_dt_nums = 300):
    events_data = EventsData()
    events_data.read_IEBCS_events(os.path.join(event_path,"raw.dat"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]
    for idx in range(0,total_dt_nums):
        img = events_data.display_events(ev_data,dt*idx,dt*(idx+1))
        cv2.imwrite(os.path.join(event_path, '{0:05d}'.format(idx) + ".png"), img)