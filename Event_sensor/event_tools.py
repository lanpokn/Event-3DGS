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
from tqdm import tqdm
from src.event_display import EventDisplay
from src.example_EXR_to_events import View_3D
def Nlerp(a1,a2,alpha):
    return alpha * a1 + (1 - alpha) *a2
    
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

#images is a tensor file
def simulate_event_camera(images,ev_full,dt=2857,lat=100, jit=10, ref=100, tau=300, th=0.3, th_noise=0.01):
    dsi.initSimu(images[0].shape[0], images[0].shape[1])
    dsi.initLatency(lat, jit, ref, tau)
    dsi.initContrast(th, th, th_noise)
    init_bgn_hist_cpp("./Event_sensor/data/noise_neg_161lux.npy", "./Event_sensor/data/noise_neg_161lux.npy")
    isInit = False
    time = 0
    ed = EventDisplay("Events",  images[0].shape[1], images[0].shape[0], dt*2)
    for im in tqdm(images, desc="generating events", unit="frame"):
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
                         10000000)
            ed.update(ev, dt)
            ev_full.increase_ev(ev)
            time += dt
    return ev_full

def save_event_result(ev_full,event_path):
    file_path = os.path.join(event_path,"raw.dat")
    ev_full.write(file_path)
    return file_path
    
def generate_images(event_path,dt, total_dt_nums):
    events_data = EventsData()
    events_data.read_IEBCS_events(os.path.join(event_path,"raw.dat"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]
    for idx in range(0,total_dt_nums):
        img = events_data.display_events(ev_data,dt*idx,dt*(idx+1))
        cv2.imwrite(os.path.join(event_path, '{0:05d}'.format(idx) + ".png"), img)
def generate_images_accumu(event_path,dt, total_dt_nums):
    events_data = EventsData()
    events_data.read_IEBCS_events(os.path.join(event_path,"raw.dat"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu(ev_data,dt*idx,dt*(idx+1))
        cv2.imwrite(os.path.join(event_path+"_ac", '{0:05d}'.format(idx+3) + ".png"), img)
def generate_images_accumu_volt(event_path,dt, total_dt_nums):
    events_data = EventsData()
    events_data.read_Volt_events(os.path.join(event_path,"raw.dat"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu(ev_data,dt*idx,dt*(idx+1))
        cv2.imwrite(os.path.join(event_path+"_ac", '{0:05d}'.format(idx+3) + ".png"), img)
def generate_images_accumu_edslike(event_path,dt, total_dt_nums,frac = 0.1):
    events_data = EventsData()
    events_data.read_IEBCS_events(os.path.join(event_path,"raw.dat"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]

    width = events_data.width
    height = events_data.height

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu(ev_data,dt*idx,dt*(idx+frac),width, height)
        cv2.imwrite(os.path.join(event_path+"/images_simu", 'frame_{0:010d}'.format(idx*10) + ".png"), img)
def generate_images_accumu_eds(event_path,dt, total_dt_nums,frac = 0.1):
    events_data = EventsData()
    events_data.read_eds_events(os.path.join(event_path,"events.h5"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]

    width = events_data.width
    height = events_data.height

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu(ev_data,dt*idx,dt*(idx+frac),width, height)
        cv2.imwrite(os.path.join(event_path+"/images_ac", 'frame_{0:010d}'.format(idx*10) + ".png"), img)
def generate_images_eds(event_path,dt, total_dt_nums,width=None, height=None):
    events_data = EventsData()
    events_data.read_eds_events(os.path.join(event_path,"events.h5"), (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]
    if width == None:
        width = events_data.width
        height = events_data.height
    for idx in range(0,total_dt_nums):
        img = events_data.display_events(ev_data,dt*idx,dt*(idx+0.2))
        cv2.imwrite(os.path.join(event_path+"/images_ev", 'frame_{0:010d}'.format(idx*10) + ".png"), img)
def generate_images_accumu_T(event_path,dt, total_dt_nums,frac = 0.1):
    events_data = EventsData()
    events_data.read_eds_events(event_path+".h5", (total_dt_nums+1)*dt)
    ev_data = events_data.events[0]

    width = events_data.width
    height = events_data.height

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu(ev_data,dt*idx,dt*(idx+frac),width, height)
        cv2.imwrite(os.path.join(event_path+"/images_ac", '{:04d}.{}'.format(idx, "png.png")), img)
def generate_images_accumu_Tumvie(event_path,dt, total_dt_nums,frac = 0.1):
    events_data = EventsData()
    ts, x, y, p =  events_data.read_Tumvie_events(event_path+".h5", (total_dt_nums)*dt)
    # ev_data = events_data.events[0]

    width = events_data.width
    height = events_data.height

    # point_cloud = events_data.display_events_3D(ev_data,0,5000)
    # View_3D(point_cloud)
    
    for idx in range(0,total_dt_nums):
        img = events_data.display_events_accumu_raw(x,y,ts,p,dt*idx,dt*(idx+frac),width, height)
        cv2.imwrite(os.path.join(event_path+"/images_ac", '{:05d}.{}'.format(idx, ".png")), img)