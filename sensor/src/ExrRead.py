import OpenEXR
import Imath
import numpy as np
import cv2
import time

#BrightScale 1 not change ,1000 muti 1000, because scene is easy, and exr has a wide range 
# we can simply muti a number to get high img based on low img, so we only need to render low img
# which can barely not seen
def read_exr_channel(exr_file,channel_name,BrightScale):
    # Read the EXR file
    exr = OpenEXR.InputFile(exr_file)

    # Get the image header and channel information
    header = exr.header()
    channel_info = header['channels']

    # Get the size of the image
    width = header['dataWindow'].max.x + 1
    height = header['dataWindow'].max.y + 1

    # Read the specified channel
    #float is between 0 and 1
    channel_data = exr.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))

    # Convert the channel data to a NumPy array
    channel_list = np.frombuffer(channel_data, dtype=np.float32)
    channel_list = np.reshape(channel_list, (height, width))

    # Create a CV data type with the same size as the channel data
    cv_image = np.zeros((height, width), dtype=np.float32)

    # Assign the channel data to the CV image
    cv_image[:, :] = channel_list * BrightScale

    # Display the CV image
    # cv2.imshow("Channel Image", cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv_image

def calculate_intensity_from_spetral(exr_file, channel_number, BrightScale):
    # Read the EXR file
    start_time = time.time()
    exr = OpenEXR.InputFile(exr_file)
    # Get the image header and channel information
    header = exr.header()
    channel_info = header['channels']
    # Get the size of the image
    width = header['dataWindow'].max.x + 1
    height = header['dataWindow'].max.y + 1

    # Create an empty list to store the channel data
    channel_list = []

    # Read each channel individually
    # here I read all 32 channel for arbitary N
    # In the view of algorithm analysis, reading channel is decided by chanel number!
    #for i in range(31, 0, -1):
    for i in range(1, 32):
        channel_name = f"Radiance.C{i:02d}"
        channel_data = exr.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
        temp = np.frombuffer(channel_data, dtype=np.float32)
        temp = np.reshape(temp, (height, width))
        channel_list.append(temp)

    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time of N hyper", total_time*channel_number/32)
    # Create a CV data type with the same size as the channel data
    cv_image = np.zeros((height, width), dtype=np.float32)
    # cv2.waitKey()
    # Assign the channel data to the CV image with weighted sum
    if channel_number == 31:
        cv_image[:, :] = (
            0.82 * channel_list[0] + 0.85 * channel_list[1] + 0.87 * channel_list[2] + 0.88 * channel_list[3] + 0.92 * channel_list[4] +
            0.95 * channel_list[5] + 0.96 * channel_list[6] + 0.96 * channel_list[7] + 0.98 * channel_list[8] + 1 * channel_list[9] + 0.99 * channel_list[10] +
            1 * channel_list[11] + 0.99 * channel_list[12] + 1 * channel_list[13] + 0.99 * channel_list[14] + 1 * channel_list[15] +
            0.99 * channel_list[16] + 0.98 * channel_list[17] + 0.98 * channel_list[18] + 0.97 * channel_list[19] + 0.95 * channel_list[20] +
            0.94 * channel_list[21] + 0.92 * channel_list[22] + 0.92 * channel_list[23] + 0.87 * channel_list[24] + 0.86 * channel_list[25] +
            0.85 * channel_list[26] + 0.82 * channel_list[27] + 0.79 * channel_list[28] + 0.78 * channel_list[29] + 0.76 * channel_list[30]
        )/31.0
    elif channel_number == 16:
        cv_image[:, :] =  (
            0.82 * channel_list[0] + 0.87 * channel_list[2] + 0.92 * channel_list[4] +
            0.96 * channel_list[6] + 0.98 * channel_list[8] + 0.99 * channel_list[10] +
            1 * channel_list[12] + 0.99 * channel_list[14] +
            0.99 * channel_list[16] + 0.98 * channel_list[18] + 0.97 * channel_list[20] +
            0.94 * channel_list[22] + 0.87 * channel_list[24] +
            0.85 * channel_list[26] + 0.79 * channel_list[28] + 0.76 * channel_list[30]
        )/16.0
    elif channel_number == 11:
        cv_image[:, :] =  (
            0.82 * channel_list[0] + 0.88 * channel_list[3] + 0.95 * channel_list[6] + 0.96 * channel_list[9] +
            1 * channel_list[12] + 1 * channel_list[15] + 0.99 * channel_list[18] +
            0.99 * channel_list[21] + 0.94 * channel_list[24] +
            0.85 * channel_list[27] + 0.78 * channel_list[30]
        )/11.0
    elif channel_number == 7:
        cv_image[:, :] =  (
            0.82 * channel_list[0] + 0.92 * channel_list[5] + 1 * channel_list[10] + 0.99 * channel_list[15] +
            0.99 * channel_list[20] + 0.92 * channel_list[25] + 0.85 * channel_list[30]
        )/7.0
    elif channel_number == 6:
        cv_image[:, :] =  (
            0.82 * channel_list[0] + 0.95 * channel_list[6] + 1 * channel_list[12] +
            0.99 * channel_list[18] + 0.94 * channel_list[24] + 0.85 * channel_list[30]
        )/6.0
    # Scale the CV image
    cv_image[:, :] = cv_image * BrightScale
    # cv_image /= 31.0
    return cv_image
# # Example usage
# exr_file = "C:/Users//Desktop/exr_test/Rotate_360_pbrt00121exr"
# channel_name = "intensity"
# read_exr_channel(exr_file, channel_name)