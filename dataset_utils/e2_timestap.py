def delete_last_string(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 将每一行按空格分割
            parts = line.split()
            # 提取数字部分并乘以 1e6，然后转换为整数
            number = int(float(parts[-1]) * 1e6)
            # 将数字写入输出文件
            f_out.write(str(number) + '/n')
            
def swap_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 将每一行按空格分割
            parts = line.split()
            # 交换位置并写入输出文件
            f_out.write(parts[1] + ' ' + parts[0] + '/n')
def rename_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 将每一行按空格分割
            parts = line.split('/')
            # 提取文件名部分
            filename = parts[-1]
            # 提取字符串中的数字部分
            number_str = filename.split('_')[-1].split('.')[0]
            # 将数字部分转换为整数
            number = int(number_str)
            # 构建新的字符串格式
            new_filename = '{:05d}.png'.format(number)
            new_line = '/'.join(parts[:-1]) + '/' + new_filename
            # 写入输出文件
            f_out.write(new_line +" "+ filename.split(' ')[-1])
import os
import shutil

def rename_images_in_txt(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for index, line in enumerate(lines):
            parts = line.split()
            new_image_name = f"images/{index:05d}.png"
            number = parts[1]
            f.write(f"{new_image_name} {number}\n")

# 示例用法
source_folder = 'D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_bad.txt'
destination_folder = 'D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps.txt'
rename_images_in_txt(source_folder, destination_folder)
# # 指定输入文件路径
# input_file = 'D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_old.txt'
# # 指定输出文件路径
# output_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_old2.txt"

# # 调用函数进行处理
# swap_lines(input_file, output_file)
# # 指定输入文件路径
# input_file = 'D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_old2.txt'
# # 指定输出文件路径
# output_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps.txt"

# # 调用函数进行处理
# rename_lines(input_file, output_file)
# # 指定输入文件路径
# input_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_old2.txt"
# # 指定输出文件路径
# output_file = "D:/2024/3DGS/PureEventFilter/data/dynamic_high_colmap_easy/image_timestamps_e2.txt"

# # 调用函数进行处理
# delete_last_string(input_file, output_file)