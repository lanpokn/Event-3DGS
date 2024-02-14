import cv2
import os
import sys
import numpy as np

# 添加路径，以便导入自定义模块
sys.path.append("../../src")

from event_buffer import EventBuffer
from dvs_sensor import init_bgn_hist_cpp, DvsSensor
from event_display import EventDisplay
import dsi

# 输入文件夹路径
input_folder = "D:/2024/3DGS/Nerf_Event/data/eds/09_ziggy_flying_pieces/images_ori"

# 参数设置
lat = 100
jit = 10
ref = 100
tau = 300
th = 0.3
th_noise = 0.01
dt = 13513


# 初始化事件缓冲区和事件显示器
ev_full = EventBuffer(1)
# ed = EventDisplay("Events", 1920, 1080, dt*2)

# 获取文件夹中的所有图片文件
image_files = sorted(os.listdir(input_folder))

# 初始化视频写入器
time = 0
isInit = False
# 遍历文件夹中的每张图片
for image_file in image_files:
    # 构造完整的图片文件路径
    image_path = os.path.join(input_folder, image_file)
    
    # 读取图片
    im = cv2.imread(image_path)
    
    
    # 将图片转换为灰度图像
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
    
    # 初始化或更新 DSI
    if not isInit:
        dsi.initSimu(int(im.shape[0]), int(im.shape[1]))  # 假设输入图片的分辨率为 1080x1920
        dsi.initLatency(lat, jit, ref, tau)
        dsi.initContrast(th, th, th_noise)
        init_bgn_hist_cpp("D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy", "D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_neg_161lux.npy")
        dsi.initImg(im)
        ed = EventDisplay("Events", int(im.shape[1]), int(im.shape[0]), dt*2)

        isInit = True
    else:
        buf = dsi.updateImg(im, dt)
        ev = EventBuffer(1)
        ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                     np.array(buf["x"], dtype=np.uint16),
                     np.array(buf["y"], dtype=np.uint16),
                     np.array(buf["p"], dtype=np.uint64),
                     10000000)
        # ed.update(ev, dt)
        ev_full.increase_ev(ev)
        time += dt
        if time > 0.1e19:
            break
ev_full.write('raw.dat')