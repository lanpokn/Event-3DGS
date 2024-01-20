"""
Funciton: showing events to observe spatio-temporal spikes.
Author information: Jianing Li, lijianing@pku.edu.cn, Peking University, May 15th, 2018.

"""
import numpy
from event_process import event_processing
from event_process.event_processing import aer_on_off_events
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_events(events_ON, events_OFF, width = 128, height =128):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events_ON    - ON events in the increasing intensity.
        events_OFF    - OFF events in the decreasing intensity.
        width    - the width of AER sensor.
        height    - the height of AER sensor.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(events_ON[1][:] / 10 ** 6, events_ON[2][:], events_ON[3][:], c='r', label='ON')
    ax.scatter(events_OFF[1][:] / 10 ** 6, events_OFF[2][:], events_OFF[3][:], c='b', label='OFF')
    ax.set_xlabel('Timestamp(s)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    fig.suptitle('Upsampling for dynamic vision sensor')
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6,
                                 max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    ax.legend()
    plt.show()


def aer_show_events(aer_data, width = 128, height =128, length = 16384):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        aer_data    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        length    - the time length.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    events_ON, events_OFF = aer_on_off_events(aer_data)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection = '3d')
    ax.scatter(events_ON[1][:], events_ON[2][:], events_ON[3][:], c='r', label='ON', s= 6)
    ax.scatter(events_OFF[1][:], events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s= 6)

    font1 = {'family': 'Times New Roman', 'size': 20}
    font1_x = {'family': 'Times New Roman', 'size': 19}
    font2 = {'size': 13}
    ax.set_xlabel('t(us)', font1_x)
    ax.set_ylabel('x', font1)
    ax.set_zlabel('y', font1)
    ax.set_xlim([0, length])
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(0, length, 5).astype(int))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.zaxis.set_tick_params(labelsize=13)

    ax.legend(loc='upper center', bbox_to_anchor=(0.77, 0.83), prop=font2)

    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    #ax.legend(loc = 0)
    plt.show()


def show_ON_OFF_events(events, width = 128, height =128, length = 16384):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events   - the dynamic vision sensor.
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        length    - the time length.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    events_ON, events_OFF = event_processing.On_off_events(events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection = '3d')
    ax.scatter(events_ON[1][:]/1000, events_ON[2][:], events_ON[3][:], c='r', label='ON', s= 6) # events_ON[1][:]
    ax.scatter(events_OFF[1][:]/1000, events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s= 6)

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = { 'family':'Times New Roman', 'size': 17}
    font3 = {'family': 'Times New Roman', 'size': 16}
    ax.set_xlabel('\n t(ms)', font1) #us
    ax.set_ylabel('\n x', font1)
    ax.set_zlabel('\n y', font1)
    ax.set_xlim([0, 200]) # int(length/1000) 200, 35
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(0, 200, 5).astype(int)) # length
    #ax.xaxis.set_tick_params(labelsize=14) #12
    #ax.yaxis.set_tick_params(labelsize=14) #13
    ax.zaxis.set_tick_params(labelsize=17) #13
    #plt.yticks(fontname="Times New Roman")
    #plt.xticks(fontname="Times New Roman")

    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    #ax.set_zticklabels(ax.get_zticks(), font1)
    #ax.zaxis.set_ticklabels(ax.get_yticks(), font2)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")


    ax.legend(loc='upper center', bbox_to_anchor=(0.76, 0.83), prop=font3)

    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    #ax.legend(loc = 0)
    plt.show()


def show_ON_OFF_spikes(events, width = 240, height =180):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events   - the dynamic vision sensor.
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        length    - the time length.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    new_events = numpy.copy(events)
    events[0, :] = new_events[3, :]
    events[2, :] = width - new_events[2, :]
    events[3, :] = new_events[0, :]

    events_ON, events_OFF = event_processing.On_off_events(events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection = '3d')
    #ax.scatter(events_ON[3][:]/1000, events_ON[2][:], events_ON[1][:],  c='r', label='ON', s= 6) # events_ON[1][:]
    ax.scatter(events_OFF[3][:]/1000, events_OFF[2][:], events_OFF[1][:], c='limegreen', label='OFF', s= 6)

    font1 = {'family': 'Times New Roman', 'size': 20}
    font1_x = {'family': 'Times New Roman', 'size': 19}
    font2 = {'size': 13}
    ax.set_xlabel('t(ms)', font1_x) #us
    ax.set_ylabel('x', font1)
    ax.set_zlabel('y', font1)
    #ax.set_xlim([0, 200]) # int(length/1000)
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    #ax.set_xticks(numpy.linspace(0, 3, 5).astype(int)) # length
    ax.set_xticks(numpy.linspace(0, 3, 5))  # length
    ax.xaxis.set_tick_params(labelsize=14) #12
    ax.yaxis.set_tick_params(labelsize=14) #13
    ax.zaxis.set_tick_params(labelsize=14) #13

    ax.legend(loc='upper center', bbox_to_anchor=(0.77, 0.83), prop=font2)

    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    #ax.legend(loc = 0)
    plt.show()


def show_davis240_ON_OFF_spikes(events, width = 240, height =180):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events   - the dynamic vision sensor, including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        width    - the width of AER sensor.
        height    - the height of AER sensor.
        length    - the time length.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    events[2, :] = width - events[2, :]

    events_ON, events_OFF = event_processing.On_off_events(events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection = '3d')
    #ax.scatter(events_ON[1][:]/1000, events_ON[2][:], events_ON[3][:],  c='r', label='ON', s= 6) # events_ON[1][:]
    #ax.scatter(events_OFF[1][:]/1000, events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s= 6)
    ax.scatter(events_ON[1][:] / 1000, events_ON[2][:], events_ON[3][:], c='red', label='ON', s=3)  # events_ON[1][:]
    ax.scatter(events_OFF[1][:] / 1000, events_OFF[2][:], events_OFF[3][:], c='mediumblue', label='OFF', s=3)

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = {'family': 'Times New Roman', 'size': 17}
    font3 = {'family': 'Times New Roman', 'size': 16}
    ax.set_xlabel('\n t(ms)', font1)  # us
    ax.set_ylabel('\n x', font1)
    ax.set_zlabel('\n y', font1)
    #ax.set_xlim([0, 200]) # int(length/1000)
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 2).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 2).astype(int))
    ax.set_xticks(numpy.linspace(0, 4, 5)) # bicylce 3, night_drive 6, sun 4.
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    ax.zaxis.set_tick_params(labelsize=17)  # 13
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")

    #ax.legend(loc='upper center', bbox_to_anchor=(0.77, 0.83), prop=font2)

    plt.show()




def show_simulating_events(events, width = 128, height =128):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events   - the dynamic vision sensor.
        width    - the width of AER sensor.
        height    - the height of AER sensor.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    events_ON, events_OFF = event_processing.On_off_events(events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection = '3d')
    ax.scatter(events_ON[1][:]/10000, events_ON[2][:], events_ON[3][:], c='r', label='ON', s= 6) # events_ON[1][:]
    ax.scatter(events_OFF[1][:]/10000, events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s= 6)

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = {'family': 'Times New Roman', 'size': 17}
    font3 = {'family': 'Times New Roman', 'size': 16}
    ax.set_xlabel('\n t(ms)', font1)  # us
    ax.set_ylabel('\n x', font1)
    ax.set_zlabel('\n y', font1)
    ax.set_xlim([0, 1]) # int(length/1000)
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(0, 1, 5)) # length
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    ax.zaxis.set_tick_params(labelsize=17)  # 13
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")

    ax.legend(loc='upper center', bbox_to_anchor=(0.76, 0.83), prop=font3)

    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    #ax.legend(loc = 0)
    plt.show()



def show_spatio_temporal_events(events, width=346, height=240):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        events   - the dynamic vision sensor.
        width    - the width of AER sensor.
        height    - the height of AER sensor.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    events = numpy.transpose(events)
    new_events = numpy.copy(events)
    new_events[0, :] = events[3, :]
    new_events[1, :] = events[0, :]
    new_events[2, :] = events[1, :]
    new_events[3, :] = events[2, :]
    events_ON, events_OFF = event_processing.On_off_events(new_events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection='3d')
    ax.scatter(events_ON[1][:], events_ON[2][:], events_ON[3][:], c='r', label='ON', s=6)  # events_ON[1][:]
    ax.scatter(events_OFF[1][:], events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s=6)

    font1 = {'family': 'Times New Roman', 'size': 22}
    font2 = {'family': 'Times New Roman', 'size': 17}
    font3 = {'family': 'Times New Roman', 'size': 16}
    ax.set_xlabel('\n t(us)', font1)  # us
    ax.set_ylabel('\n x', font1)
    ax.set_zlabel('\n y', font1)
    ax.set_xlim([0, max(new_events[1][:])])  # int(length/1000)
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(0, max(new_events[1][:]), 5).astype(int))  # length
    ax.set_xticklabels(ax.get_xticks(), font2)
    ax.set_yticklabels(ax.get_yticks(), font2)
    ax.zaxis.set_tick_params(labelsize=17)  # 13
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontname("Times New Roman")
    ax.legend(loc='upper center', bbox_to_anchor=(0.76, 0.83), prop=font2)


    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    # ax.legend(loc = 0)
    plt.show()


def show_noise_events(true_events, events, width=346, height=240):
    """
    plot events in three-dimensional space.

    Inputs:
    -------
        true events   - the true event signal.
        events   - events include true event signal and noise events.
        width    - the width of AER sensor.
        height   - the height of AER sensor.

    Outputs:
    ------
        figure     - a figure shows events in three-dimensional space.

    """
    index = numpy.array(numpy.all((events[:, None, 1:] == true_events[None, :, 1:]), axis=-1).nonzero()).T.tolist()
    noise_index = numpy.array(index)

    true_events = numpy.transpose(true_events)
    events = numpy.transpose(events)
    noise_events = numpy.copy(events)

    for i in range(events.shape[1]):
        for j in range(true_events.shape[1]):

            if numpy.array_equal(events[1:, i], true_events[1:, j]):
                numpy.delete(noise_events, i, 1)

    new_events = numpy.copy(true_events)
    new_events[0, :] = true_events[3, :]
    new_events[1, :] = true_events[0, :]
    new_events[2, :] = true_events[1, :]
    new_events[3, :] = true_events[2, :]
    events_ON, events_OFF = event_processing.On_off_events(new_events)
    fig = plt.figure('{} * {}'.format(width, height))
    ax = fig.gca(projection='3d')
    ax.scatter(events_ON[1][:], events_ON[2][:], events_ON[3][:], c='r', label='ON', s=6)  # events_ON[1][:]
    ax.scatter(events_OFF[1][:], events_OFF[2][:], events_OFF[3][:], c='limegreen', label='OFF', s=6)

    font1 = {'family': 'Times New Roman', 'size': 20}
    font1_x = {'family': 'Times New Roman', 'size': 19}
    font2 = {'size': 13}
    ax.set_xlabel('t(us)', font1_x)  # us
    ax.set_ylabel('x', font1)
    ax.set_zlabel('y', font1)
    ax.set_xlim([0, max(new_events[1][:])])  # int(length/1000)
    ax.set_ylim([1, width])
    ax.set_zlim([1, height])
    # print(numpy.linspace(1, width, 5).astype(int))
    ax.set_yticks(numpy.linspace(1, width, 5).astype(int))
    ax.set_zticks(numpy.linspace(1, height, 5).astype(int))
    ax.set_xticks(numpy.linspace(0, max(new_events[1][:]), 5))  # length
    ax.xaxis.set_tick_params(labelsize=14)  # 12
    ax.yaxis.set_tick_params(labelsize=14)  # 13
    ax.zaxis.set_tick_params(labelsize=14)  # 13

    ax.legend(loc='upper center', bbox_to_anchor=(0.77, 0.83), prop=font2)

    # ax.set_xticks(numpy.linspace(min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6, 5))

    # ax.set_ylim([0, width])
    # ax.set_zlim([0, height])
    # ax.set_xlim([min(min(events_ON[1][:]), min(events_OFF[1][:])) / 10 ** 6, max(max(events_ON[1][:]), max(events_OFF[1][:])) / 10 ** 6])
    # ax.legend(loc = 0)
    plt.show()