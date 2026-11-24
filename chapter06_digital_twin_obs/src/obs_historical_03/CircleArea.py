import numpy as np


# def get_Endpoint(point1, point2):
#     # 计算直线方程 y = mx + b
#     if point2[0] - point1[0] != 0:
#         m = (point2[1] - point1[1]) / (point2[0] - point1[0])
#         b = point1[1] - m * point1[0]
#     else:
#         m = float('inf')  # 处理斜率无穷大的情况
#         b = point1[0]  # 垂直线的 x 截距

#     # 计算两点之间的距离
#     d = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
#     # 计算直线上距离 point1 有 (d + 10) 的点
#     if m != float('inf'):
#         x = round(point1[0] + (d + 10) / np.sqrt(1 + m**2))
#         y = round(m * x + b)
#     else:
#         x = point1[0]
#         y = round(point1[1] + d + 10)

#     # 计算两个点的距离
#     np2 = [x, y]
#     distance_p1_np2 = round(np.sqrt((np2[0] - point1[0])**2 + (np2[1] - point1[1])**2))
    
#     # 计算圆的中心和半径
#     center_x = round((point1[0] + np2[0]) / 2)
#     center_y = round((point1[1] + np2[1]) / 2)
#     radius = round(distance_p1_np2 / 2)

#     return center_x, center_y, radius

def get_Endpoint(point1, point2):
    # 计算直线方程 y = mx + b
    if point2[0] - point1[0] != 0:
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - m * point1[0]
    else:
        m = float('inf')  # 处理斜率无穷大的情况
        b = point1[0]  # 垂直线的 x 截距

    # 计算两点之间的距离
    d = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    # 计算直线上距离 point2 有 (d + 10) 的两个点
    if m != float('inf'):
        x1 = round(point2[0] + (d + 30) / np.sqrt(1 + m**2))
        y1 = round(m * x1 + b)
        x2 = round(point2[0] - (d + 30) / np.sqrt(1 + m**2))  # 另一个交点
        y2 = round(m * x2 + b)
    else:
        x1 = point2[0]
        y1 = round(point2[1] + d + 30)
        x2 = point2[0]
        y2 = round(point2[1] - (d + 30))  # 另一个交点

    # 计算两个点与 point1 和 point2 的距离
    distance1_p1 = np.sqrt((x1 - point1[0])**2 + (y1 - point1[1])**2)
    distance2_p1 = np.sqrt((x2 - point1[0])**2 + (y2 - point1[1])**2)
    distance1_p2 = np.sqrt((x1 - point2[0])**2 + (y1 - point2[1])**2)
    distance2_p2 = np.sqrt((x2 - point2[0])**2 + (y2 - point2[1])**2)

    # 选择离 point1 更远的那个点
    if distance1_p1 > distance2_p1:
        x = x1
        y = y1
    else:
        x = x2
        y = y2


    # 计算两个点的距离
    np2 = [x, y]
    distance_p1_np2 = round(np.sqrt((np2[0] - point1[0])**2 + (np2[1] - point1[1])**2))
    
    # 计算圆的中心和半径
    center_x = round((point1[0] + np2[0]) / 2)
    center_y = round((point1[1] + np2[1]) / 2)
    radius = round(distance_p1_np2 / 2)

    return center_x, center_y, radius

# 修改 extract_circle_region 函数
def extract_circle_region(grid_map, point1, point2):
    center_x, center_y, radius = get_Endpoint(point1, point2)  # 接收三个返回值
    h, k = center_x, center_y
    r_squared = radius**2
    circle_region = []

    for x in range(len(grid_map)):
        for y in range(len(grid_map[0])):
            if (x - h)**2 + (y - k)**2 <= r_squared:
                circle_region.append((x, y))

    # print('circle_region', circle_region)

    return circle_region

# def get_valid_coordinates(circleRegion, gridMap):
#     validCords = []
#     for x, y in circleRegion:
#         if gridMap[x, y] == 0:
#             validCords.append((x, y))
#     return validCords


# def sample(gridMap, point1, point2):
#     # 获取圆内的有效区域
#     circleRegion = extract_circle_region(gridMap, point1, point2)

#     # 获取有效区域的坐标
#     validRegion = get_valid_coordinates(circleRegion, gridMap)

#     # 如果可选样本不足 10 个，直接返回全部可用样本
#     if len(validRegion) < 10:
#         return validRegion

#     # 计算每个子区域的大小
#     num_samples = len(validRegion)
#     samples_per_region = num_samples // 10

#     # 从每个子区域中各选择一个点
#     random_samples = []
#     for i in range(10):
#         start_index = i * samples_per_region
#         end_index = (i + 1) * samples_per_region
#         region_samples = validRegion[start_index:end_index]
#         if region_samples:
#             random_samples.append(random.choice(region_samples))

#     return random_samples

# #测试get_Endpoint
# point1 = [1, 2]
# point2 = [2, 2]

# center_x, center_y, radius = get_Endpoint(point1, point2)
# print(f"圆心坐标: ({center_x}, {center_y}), 半径: {radius}")

# # 测试extract_circle_region
# def test_extract_circle_region():
#     # 示例的 grid_map，包含 0 和 1 的值
#     grid_map = [
#         [1, 0, 1, 0, 1],
#         [0, 1, 1, 0, 0],
#         [1, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1],
#         [1, 1, 1, 0, 1]
#     ]

#     # 选择两个点作为直径
#     point1 = [0, 0]
#     point2 = [2,2 ]

#     # 调用 extract_circle_region 函数
#     circle_region = extract_circle_region(grid_map, point1, point2)

#     # 输出结果
#     print('圆内的有效区域：', circle_region)
    
# # 调用测试函数
# test_extract_circle_region()

# import cv2

# def create_test_grid_map():
#     # 创建一个 10x10 的网格地图，1 表示障碍物，0 表示可通行
#     grid_map = np.zeros((100, 100), dtype=np.uint8)
#     grid_map[2:8, 2:8] = 1  # 设置中间一块区域为障碍物
#     return grid_map

# # 创建测试网格地图
# grid_map = create_test_grid_map()

# # 定义两点坐标用于测试
# point1 = [1, 1]  # 位于障碍物之外
# point2 = [50, 58]  # 位于障碍物之外

# # 测试 get_Endpoint 函数
# def test_get_Endpoint(point1, point2):
#     center_x, center_y, radius = get_Endpoint(point1, point2)
#     print("Center: ({}, {})".format(center_x, center_y))
#     print("Radius: {}".format(radius))

# test_get_Endpoint(point1, point2)

# # 绘制地图和圆圈
# map_copy = grid_map.copy()
# map_image = np.zeros((map_copy.shape[0], map_copy.shape[1], 3), dtype=np.uint8)
# map_image[map_copy == 1] = (0, 0, 0)
# map_image[map_copy == 0] = (255, 255, 255)
# cv2.circle(map_image, (point1[1], point1[0]), 10, (0, 0, 255), -1)
# cv2.circle(map_image, (point2[1], point2[0]), 10, (0, 255, 0), -1)

# # 绘制圆圈
# center_x, center_y, radius = get_Endpoint(point1, point2)
# cv2.circle(map_image, (center_y, center_x), radius, (0, 0, 0), 2)

# cv2.imshow("Test Grid Map with Circles", map_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import random
# from Image2gridMap import create_grid_map


# grid_map = create_grid_map('grid_map.png')
# # 示例的 gridMap，point1 和 point2

# point1 = [150, 50] #例子start point
# point2 = [120, 250] #例子goal point


# # 测试 get_Endpoint 函数
# def test_get_Endpoint(point1, point2):
#     center_x, center_y, radius = get_Endpoint(point1, point2)
#     print("Center: ({}, {})".format(center_x, center_y))
#     print("Radius: {}".format(radius))

# test_get_Endpoint(point1, point2)

# # 绘制地图和圆圈
# map_copy = grid_map.copy()
# map_image = np.zeros((map_copy.shape[0], map_copy.shape[1], 3), dtype=np.uint8)
# map_image[map_copy == 1] = (0, 0, 0)
# map_image[map_copy == 0] = (255, 255, 255)

# # 绘制圆圈
# center_x, center_y, radius = get_Endpoint(point1, point2)
# cv2.circle(map_image, (center_y, center_x), radius, (0, 0, 0), 2)

# # 随机选择的点用红色圆圈表示
# samples = sample(grid_map, point1, point2)
# for sample in samples:
#     cv2.circle(map_image, (sample[1], sample[0]), 5, (0, 0, 255), -1)

# # 绘制 p1 和 p2
# cv2.circle(map_image, (point1[1], point1[0]), 10, (255, 0, 0), -1)
# cv2.drawMarker(map_image, (point2[1], point2[0]), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

# cv2.imshow("Test Grid Map with Circles and Points", map_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()