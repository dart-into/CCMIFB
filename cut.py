import os
import cv2
import numpy as np


def process_images_in_folder(input_image_path):
    # 读取图像
    image = cv2.imread(input_image_path)

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # 寻找图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储所有轮廓的边界值
    all_boundaries = []

    # 循环处理每个轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        all_boundaries.append((x, y, x + w, y + h))

    # 找到包含所有轮廓的大矩形的边界值
    x_min = min([x for (x, _, _, _) in all_boundaries])
    y_min = min([y for (_, y, _, _) in all_boundaries])
    x_max = max([x_max for (_, _, x_max, _) in all_boundaries])
    y_max = max([y_max for (_, _, _, y_max) in all_boundaries])

    # 找到大矩形的宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 找到边界值的最大范围（最大的宽度和高度）
    max_range = max(width, height)


    # 计算正方形的四个边界值
    new_x_min = max(0, x_min - (max_range - width) // 2)
    new_y_min = max(0, y_min - (max_range - height) // 2)
    new_x_max = min(image.shape[1], new_x_min + max_range)
    new_y_max = min(image.shape[0], new_y_min + max_range)

    # 裁剪图像以获得扩展后的正方形区域
    cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
    # 覆盖原始图像文件
    cv2.imwrite(input_image_path, cropped_image)


# 调用函数并处理文件夹中的图像
main_folder = "/home/user/dataset3"
#process_images_in_folder(input_folder)

import os

# 获取主文件夹中的所有子文件夹
subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

# 处理每个子文件夹中的图片
for subfolder in subfolders:
    # 获取子文件夹中的所有图片文件
    image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        # 打开图片
        process_images_in_folder(image_file)
