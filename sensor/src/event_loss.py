from event_file_io import EventsData
import numpy as np
from scipy.spatial import KDTree
import math
import time
import cv2

##loss 1
#do not use firt 2000 data!
def normalize_evs(evs):
    """normalize evs data

    Args:
        evs (np array):events data1, should be the same as the format of EventsData().events[i]

    Returns:
        evs_new: data after normalize
    """    
    evs_normalized = evs.copy()
    epsilon = 1e-8  # Small epsilon value to avoid division by zero
    evs_normalized['x'] = ((evs['x'] - evs['x'].min()) * 100 / (evs['x'].max() - evs['x'].min() + epsilon)).astype(float)
    evs_normalized['y'] = ((evs['y'] - evs['y'].min()) * 100 / (evs['y'].max() - evs['y'].min() + epsilon)).astype(float)
    evs_normalized['p'] = ((evs['p'] - evs['p'].min()) * 100 / (evs['p'].max() - evs['p'].min() + epsilon)).astype(float)
    evs_normalized['t'] = ((evs['t'] - evs['t'].min()) * 1000 / (evs['t'].max() - evs['t'].min() + epsilon)).astype(float)
    # evs_normalized['x'] = ((evs['x'] - evs['x'].min()) )/ (1.00)
    # evs_normalized['y'] = ((evs['y'] - evs['y'].min()) )/ (1.00)
    # evs_normalized['p'] = ((evs['p'] - evs['p'].min()) )* 200.00
    # evs_normalized['t'] = ((evs['t'] - evs['t'].min()) ) / (1000.00)#t: ms
    # evs_normalized['t'] = ((evs['t'] - evs['t'].min()))
    return evs_normalized

def chamfer_distance_loss(evs1, evs2):
    """use chamfer_distance to calculate the loss between two events data

    Args:
        evs1 (dictionary): events data1, should be the same as the format of EventsData().events[i]
        evs2 (dictionary): events data2, should be the same as the format of EventsData().events[i]

    Returns:
        chamfer distance
    """
    
    evs1_norm = normalize_evs(evs1)
    evs2_norm = normalize_evs(evs2)
    
    evs1_float = np.zeros((evs1_norm.shape[0], 4), dtype=np.float64)
    evs2_float = np.zeros((evs2_norm.shape[0], 4), dtype=np.float64)
    
    evs1_float[:, 0] = evs1_norm['x']
    evs1_float[:, 1] = evs1_norm['y']
    evs1_float[:, 2] = evs1_norm['p']
    evs1_float[:, 3] = evs1_norm['t']
    
    evs2_float[:, 0] = evs2_norm['x']
    evs2_float[:, 1] = evs2_norm['y']
    evs2_float[:, 2] = evs2_norm['p']
    evs2_float[:, 3] = evs2_norm['t']
    
    # Create KDTree using evs2_float as points
    tree1 = KDTree(evs1_float)
    tree2 = KDTree(evs2_float)

    # Query the tree with evs1_float as points
    # use log to avoid some point too far away
    dists1, _ = tree2.query(evs1_float)
    # dists1 = np.log(10*dists1+1)
    dists2, _ = tree1.query(evs2_float)
    # dists2 = np.exp(10*dists2+1)

    # Return the sum of distances
    return (np.mean(dists1)+np.mean(dists2))

def gaussian_distance_loss(evs1, evs2,sigma = 0.4):
    """use chamfer_distance to calculate the loss between two events data

    Args:
        evs1 (dictionary): events data1, should be the same as the format of EventsData().events[i]
        evs2 (dictionary): events data2, should be the same as the format of EventsData().events[i]

    Returns:
        chamfer distance
    """
    
    evs1_norm = normalize_evs(evs1)
    evs2_norm = normalize_evs(evs2)
    
    evs1_float = np.zeros((evs1_norm.shape[0], 4), dtype=np.float64)
    evs2_float = np.zeros((evs2_norm.shape[0], 4), dtype=np.float64)
    
    evs1_float[:, 0] = evs1_norm['x']
    evs1_float[:, 1] = evs1_norm['y']
    evs1_float[:, 2] = evs1_norm['p']
    evs1_float[:, 3] = evs1_norm['t']
    
    evs2_float[:, 0] = evs2_norm['x']
    evs2_float[:, 1] = evs2_norm['y']
    evs2_float[:, 2] = evs2_norm['p']
    evs2_float[:, 3] = evs2_norm['t']
    
    # Create KDTree using evs2_float as points
    tree1 = KDTree(evs1_float)
    tree2 = KDTree(evs2_float)

    # Query the tree with evs1_float as points
    # use log to avoid some point too far away
    dists1, _ = tree2.query(evs1_float)
    dists1 = 1 - np.exp(-dists1 * dists1 / sigma)  # modify the calculation f

    dists2, _ = tree1.query(evs2_float)
    dists2 = 1 - np.exp(-dists2 * dists2 / sigma)  # modify the calculation for dists2

    # dists2 = np.exp(10*dists2+1)

    # Return the sum of distances
    return (np.mean(dists1)+np.mean(dists2))

def cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma):
    """
    extern
    
    Computing inner product between spike cubes using 3d gaussian kernel method.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.

    Outputs:
    -------
        inner_product    - the inner product between events and new_events.,has no upper limits

    """
    #print('events number={}'.format(len(events[0,:])))
    #print('ON number={}'.format(np.sum(events[0, :]==1)))
    ON_scale = np.sum(events[0, :]==1)/(len(events[0, :])) # ON events in history
    # new_OFF_scale = np.sum(new_events[0, :]==-1)/len(events[0, :]) # ON new events in history
    new_ON_scale = np.sum(new_events[0, :] == 1) / (len(new_events[0, :]))  # ON new events in history

    # print('events_numbers={}, new_events_numbers={}'.format(len(events[0, :]), len(new_events[0, :])))

    polarity_scale = ON_scale*new_ON_scale + (1-ON_scale)*(1-new_ON_scale)
    # polarity_scale = 1 + abs(ON_scale-new_OFF_scale) # simply polarity for integrated formulation.

    x_index = events[2, :][:, None] - new_events[2, :][None, :]
    y_index = events[3, :][:, None] - new_events[3, :][None, :]
    t_index = events[1, :][:, None] - new_events[1, :][None, :]

    dist_matrix = np.exp(- x_index**2 / (2*x_sigma**2) - y_index**2 / (2*y_sigma**2) - t_index**2 / (2*t_sigma**2))

    inner_product = polarity_scale * np.sum(dist_matrix)

    return inner_product


def cubes_3d_kernel_distance(events, new_events, x_sigma, y_sigma, t_sigma):
    """
    extern
    
    Computing distance between spike cubes using inner product in RKHS.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.

    Outputs:
    -------
        distance    - the distance between events and new_events.

    """

    if len(np.transpose(events)) <= 5 or len(np.transpose(new_events)) <= 5:
        distance = 0
    else:

        distance = cubes_3d_kernel_method(events, events, x_sigma, y_sigma, t_sigma)
        distance += cubes_3d_kernel_method(new_events, new_events, x_sigma, y_sigma, t_sigma)
        distance -= 2 * cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma)

    return distance

def events_to_spike_cubes(events, width, height, x_cube_size, y_cube_size, t_cube_size):
    """
    extern
    
    events are split into spike cubes.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        width, height    - the width and height resolutions of dynamic vision sensor.
        x_cube_size, y_cube_size, t_cube_size    - the width, height and temporal size of spike cubes.
   Outputs:
    -------
        events_cubes    - the cubes of events.

    """

    num = int((width/x_cube_size)*(height/y_cube_size)*(math.ceil(max(events[1, :] / t_cube_size))))
    #num = int((width/x_cube_size)*(height/y_cube_size)*(math.floor(max(events[1, :] / t_cube_size))))
    events_cube = [[] for _ in range(num)]
    #print('num={}'.format(num))

    for i in range(events.shape[1]):

        k = math.floor(events[2, i]/x_cube_size) + math.floor(events[3, i]/y_cube_size)*int(width/x_cube_size) + math.floor(events[1, i]/t_cube_size)*int(width/x_cube_size)*int(height/y_cube_size)
        # if k == 102:
        #     print(k)
        #print('t_cube_size={}, k={}, i={}, event={}, feature={}'.format(t_cube_size, k, i, events[:, i], math.floor(events[1, i]/t_cube_size)))
        events_cube[k].append(events[:, i])


    return events_cube

## this is the new loss!
def kernel_method_spike_cubes_loss(events, new_events, width=128, height=128, x_cube_size=32, y_cube_size=32, t_cube_size=5000, x_sigma=5, y_sigma=5, t_sigma=5000):
    """ 
        extern
                
        change some code to use EVK3 data as input
        
        3d gaussian kernel method  for spike cubes, such as polarity independent and polarity interference.

        Inputs:
        -------
            events    - events include polarity, timestamp, x and y.
            new_events    - events after changing operation.
            width, height  - the width and height of dynamic vision sensor.
            x_cube_size, y_cube_size, t_cube_size  - the size of spike cube.
            x_sigma, y_sigma, t_sigma  - the 3d gaussian kernel parameters.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """
    ##ADD IT TO fix dimension bug
    evs1_float = np.zeros((4,events.shape[0]), dtype=np.float64)
    evs2_float = np.zeros((4,new_events.shape[0]), dtype=np.float64)
    
    evs1_float[0, :] = events['p']
    evs1_float[1, :] = events['t']
    evs1_float[2, :] = events['x']
    evs1_float[3, :] = events['y']
    
    evs2_float[0, :] = new_events['p']
    evs2_float[1, :] = new_events['t']
    evs2_float[2, :] = new_events['x']
    evs2_float[3, :] = new_events['y']
    #t_intervel = evs2_float[1, :][-1]-evs2_float[1, :][0]+evs1_float[1, :][-1]-evs1_float[1, :][0]
    t_intervel = len(evs2_float[1, :])+len(evs1_float[1, :])
    # evs1_float = np.transpose(evs1_float)
    # evs2_float = np.transpose(evs2_float)

    events_cube = events_to_spike_cubes(evs1_float, width, height, x_cube_size, y_cube_size, t_cube_size)
    new_events_cubes = events_to_spike_cubes(evs2_float, width, height, x_cube_size, y_cube_size, t_cube_size)

    distance = 0
    for k in range(0, min(len(events_cube), len(new_events_cubes))):

        events_data = np.transpose(np.array(events_cube[k]))
        new_events_data = np.transpose(np.array(new_events_cubes[k]))

        if len(events_data)==0 and len(new_events_data)==0:
            distance += 0

        else:
            distance += cubes_3d_kernel_distance(events_data, new_events_data, x_sigma, y_sigma, t_sigma)
    #it's better to return distance/T_intervel
    return distance/t_intervel
    # change to [0  - 1]

def main():
    """use it to test functions
    """    
    events_data = EventsData()
    events_data_IEBCS = EventsData()
    #make sure the video is long enough, or it can't disolay normally
    events_data.read_real_events("D:/2023/computional imaging/my_pbrt/output/rotatea_360_highlight/High_360_120deg.hdf5", 1000000)
    events_data_IEBCS.read_IEBCS_events("D:/2023/computional imaging/my_pbrt/evrotate_60_highlight_100_10_100_300_0.3_0.01.dat", 1000000)
    
    ev_data0 = events_data.events[0]
    ev_data1 = events_data_IEBCS.events[0]
    
    start_time = time.time()
    # loss_same = kernel_method_spike_cubes_loss(ev_data0, ev_data0,events_data.width,events_data.height)
    # print(loss_same)
    # loss_different = kernel_method_spike_cubes_loss(ev_data0, ev_data1,events_data.width,events_data.height)
    # print(loss_different)
    # events_data.display_events(ev_data0,3000)
    # events_data.display_events(ev_data1,3000)
    
    # events_data.display_events(ev_data0_simi,3000)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of kernel method without kdtree: ", total_time)
    
    #chamfer
    start_time = time.time()
    # loss_same = chamfer_distance_loss(ev_data0, ev_data0)
    # print(loss_same)
    # loss_different = chamfer_distance_loss(ev_data0, ev_data1)
    # print(loss_different)
    # events_data.display_events_metavision(ev_data0,300)
    # img = events_data_IEBCS.display_events(ev_data1,0,200)
    # cv2.imshow("img",img)
    # cv2.waitKey()
    img = events_data.display_events(ev_data0,2400,3000)
    cv2.imshow("img",img)
    cv2.waitKey()
    # events_data.display_events(ev_data0_simi,3000)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of chamfer distance method: ", total_time)

    # for ev_data in events_data.events:
    #     print("----- New event buffer! -----")
    #     counter = ev_data.size
    #     min_t = ev_data['t'][0]
    #     max_t = ev_data['t'][-1]
    #     print(f"There were {counter} events in this event buffer.")
    #     print(f"There were {events_data.global_counter} total events up to now.")
    #     print(f"The current event buffer included events from {min_t} to {max_t} microseconds.")
    #     print("----- End of the event buffer! -----")

    # duration_seconds = events_data.global_max_t / 1.0e6
    # print(f"There were {events_data.global_counter} events in total.")
    # print(f"The total duration was {duration_seconds:.2f} seconds.")
    # if duration_seconds >= 1:
    #     print(f"There were {events_data.global_counter / duration_seconds:.2f} events per second on average.")


if __name__ == "__main__":
    main()