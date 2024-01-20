import numpy as np

# Load the .npy file
# this noise dose not affect blur, almost useless
# TODO, I don't understand it
data1 = np.load('D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_0.1lux.npy')

# Perform any necessary edits on the data array
# ...
# a = data[0]
# b = data[0][0]
# c = data[0][0][0]
# print(a)
# print(b)
# print(c)
data2 = np.load('D:/2023/computional imaging/my_pbrt/IEBCS-main/data/noise_pos_161lux.npy')
# Check if both arrays have the same shape
if data1.shape == data2.shape:
    # Calculate the average
    average_data = np.average([data1, data2], axis=0)
    # Save the average data to a new file
    np.save('average_161_0.1_data.npy', average_data)
    print("Average data saved successfully.")
else:
    print("Error: The shapes of data1 and data2 are not the same.")