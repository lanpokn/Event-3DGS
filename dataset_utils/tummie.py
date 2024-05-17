import os
import shutil

# def copy_last_100_images(source_folder, destination_folder):
#     # 获取源文件夹中所有图片文件的路径
#     image_files = [file for file in os.listdir(source_folder) if file.endswith('.jpg')]
#     # 获取最后100张图片的文件名
#     last_100_images = image_files[-100:]
    
#     # 遍历最后100张图片，复制到目标文件夹并重新命名
#     for index, image_file in enumerate(last_100_images):
#         source_path = os.path.join(source_folder, image_file)
#         destination_path = os.path.join(destination_folder, f"{index:05d}.jpg")  # 使用0填充的数字命名
#         shutil.copyfile(source_path, destination_path)


# def extract_last_100_lines(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()
#     last_100_lines = lines[-100:]

#     with open(output_file, 'w') as f:
#         f.writelines(last_100_lines)
def copy_last_100_images(source_folder, destination_folder):
    # 获取源文件夹中所有图片文件的路径
    image_files = [file for file in os.listdir(source_folder)]
    # 获取最后100张图片的文件名
    last_100_images = image_files[-100:]
    
    # 遍历最后100张图片，复制到目标文件夹并重新命名
    for index, image_file in enumerate(last_100_images):
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, f"{index:05d}.png")  # 使用0填充的数字命名
        shutil.copyfile(source_path, destination_path)


def extract_last_100_lines(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    last_100_lines = lines[-100:]

    with open(output_file, 'w') as f:
        f.writelines(last_100_lines)
def copy_first_100_images(source_folder, destination_folder):
    # 获取源文件夹中所有图片文件的路径
    image_files = [file for file in os.listdir(source_folder)]
    # 获取最后100张图片的文件名
    last_100_images = image_files[0:100]
    
    # 遍历最后100张图片，复制到目标文件夹并重新命名
    for index, image_file in enumerate(last_100_images):
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, f"{index:05d}.png")  # 使用0填充的数字命名
        shutil.copyfile(source_path, destination_path)


def extract_first_100_lines(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    last_100_lines = lines[0:100]

    with open(output_file, 'w') as f:
        f.writelines(last_100_lines)


# 指定源文件夹和目标文件夹的路径
source_folder = "D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/images"
destination_folder = "D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/renders"

# 指定输入文件和输出文件的路径
input_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/images.txt"
output_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/image_timestamps_old.txt"

# 调用函数提取输入文件的最后100行到输出文件中
# copy_first_100_images(source_folder, destination_folder)
# extract_first_100_lines(input_file, output_file)
copy_last_100_images(source_folder, destination_folder)
extract_last_100_lines(input_file, output_file)
def copy_lines_until_threshold(input_file, output_file, threshold):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 将每一行的第一个数字提取出来并乘以 1e6，然后转换为整数
            first_number = int(float(line.split()[0]) * 1e6)
            # 如果第一个数字小于等于阈值，则将该行写入输出文件
            if first_number <= threshold:
                # 将第一个数字重新转换为字符串，然后将整行写入输出文件
                line_parts = line.split()
                line_parts[0] = str(first_number)
                f_out.write(' '.join(line_parts) + '\n')
            # 如果第一个数字大于阈值，则跳出循环
            else:
                break
def copy_lines_after_threshold(input_file, output_file, threshold):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        copy = False
        for line in f_in:
            # 提取每行的第一个数字并乘以 1e6，然后转换为整数
            first_number = int(float(line.split()[0]) * 1e6)
            # 检查第一个数字是否大于阈值
            if first_number > threshold:
                copy = True
            # 如果当前行的第一个数字超过阈值，则开始写入输出文件
            if copy:
                # 将第一个数字重新转换为字符串，然后将整行写入输出文件
                line_parts = line.split()
                line_parts[0] = str(first_number)
                f_out.write(' '.join(line_parts) + '\n')

# 指定输入文件路径
input_file = 'D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/events.txt'
# 指定输出文件路径
output_file = 'D:/2024/3DGS/PureEventFilter/data/dynamic_translation_colmap_easy/calibration_volt.txt'
# 指定阈值
threshold = 55 * 1e6 # 你可以将阈值替换为你想要的值

# # 调用函数进行文件内容复制
# copy_lines_until_threshold(input_file, output_file, threshold)
copy_lines_after_threshold(input_file, output_file, threshold)

