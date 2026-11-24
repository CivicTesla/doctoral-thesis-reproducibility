import cv2
import numpy as np
import random




def create_grid_map(image_path):
	# 读取图像
	image = cv2.imread(image_path)

	# 将图像转换为灰度
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 设置阈值，将图像二值化，黑色区域标记为1，白色区域标记为0
	_, gridMap = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV)

	
	return gridMap


def get_valid_coordinates(circleRegion, gridMap):
    validCords = []
    for x, y in circleRegion:
        if gridMap[x, y] == 0:
            validCords.append((x, y))
    return validCords



def get_valid_from_whole_map(gridMap):
    valid_coordinates = []

    # 遍历整个gridMap
    for x in range(len(gridMap)):
        for y in range(len(gridMap[0])):
            # 如果当前格子的值为0，将其坐标加入结果列表
            if gridMap[y][x] == 0:
                valid_coordinates.append((y, x))
    

    return valid_coordinates


# #TestCode#############################################################################
# def test_valid_coordinates_display(gridMap):
#     # 读取地图图像
#     # grid_map = cv2.imread('grid_map.png', cv2.IMREAD_GRAYSCALE)
#     # grid_map = cv2.imread('grid_map.png')
#     grid_map = cv2.imread('dilatedImg4orgMap.png')

#     # 获取有效坐标
#     valid_coordinates = get_valid_from_whole_map(gridMap)

#     # 将有效坐标用红色标记在地图上
#     for coord in valid_coordinates:
#         cv2.circle(grid_map, (coord[1],coord[0]), 2, (0, 0, 255), -1)  # -1 表示填充圆

#     # 显示带有有效坐标的地图
#     cv2.imshow('Grid Map with Valid Coordinates', grid_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # 创建测试地图
# gridMap = create_grid_map('dilatedImg4orgMap.png')

# # 调用测试函数
# test_valid_coordinates_display(gridMap)
