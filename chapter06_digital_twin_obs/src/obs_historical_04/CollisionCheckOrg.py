import numpy as np
from skimage.draw import line


# def no_obstacle_on_line(grid_map, point1, point2):
#     # rr, cc = line(point1[0], point1[1], point2[0], point2[1])
#     rr, cc = line(point1.x, point1.y, point2.x, point2.y)
#     for r, c in zip(rr, cc):
#         if grid_map[r][c] == 1:
#             return False  # There is an obstacle on the line
#     return True  # No obstacles found on the line



# def no_obstacle_on_line(grid_map, point1, point2):
#     point1_int = (int(point1.x), int(point1.y))
#     point2_int = (int(point2.x), int(point2.y))
#     rr, cc = line(point1_int[0], point1_int[1], point2_int[0], point2_int[1])
#     for r, c in zip(rr, cc):
#         if grid_map[r][c] == 1:
#             return False  # There is an obstacle on the line
#     return True  # No obstacles found on the line


# from bresenham import bresenham

# def no_obstacle_on_line(grid_map, point1, point2):
#     line_coordinates = list(bresenham(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])))

#     for coord in line_coordinates:
#         r, c = coord
#         if grid_map[r][c] == 1:
#             return False  # There is an obstacle on the line

#     return True 


# #original Obstacle Avoidance Code
# def no_obstacle_on_line(grid_map, point1, point2):
#     point1_int = (int(point1[0]), int(point1[1]))
#     point2_int = (int(point2[0]), int(point2[1]))
#     rr, cc = line(point1_int[0], point1_int[1], point2_int[0], point2_int[1])
#     for r, c in zip(rr, cc):
#         if grid_map[r][c] == 1:
#             return False  # There is an obstacle on the line
#     return True  # No obstacles found on the line



# def no_obstacle_on_line(grid_map, point1, point2):
#     # point1_int = (int(point1[0]), int(point1[1]))
#     # point2_int = (int(point2[0]), int(point2[1]))
#     # point1_int = point1
#     # point2_int = point2

#     # rr, cc = line(point1_int[0], point1_int[1], point2_int[0], point2_int[1])
#     rr, cc = line(point1[0], point1[1], point2[0], point2[1])

#     for r, c in zip(rr, cc):
#         if not (0 <= r < len(grid_map) and 0 <= c < len(grid_map[0])):
#             continue  # Skip coordinates outside the grid

#         if grid_map[r][c] == 1:
#             return False  # There is an obstacle on the line

#     return True  # No obstacles found on the line


# Bresenham's Line Algorithm
def no_obstacle_on_line(grid_map, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if x1 < x2:
        sx = 1
    else:
        sx = -1

    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while x1 != x2 or y1 != y2:
        if not (0 <= x1 < len(grid_map) and 0 <= y1 < len(grid_map[0])):
            return False  # 超出地图范围，有障碍物

        if grid_map[x1][y1] == 1:
            return False  # 有障碍物

        e2 = 2 * err

        if e2 > -dy:
            err = err - dy
            x1 = x1 + sx

        if e2 < dx:
            err = err + dx
            y1 = y1 + sy

    return True  # 没有障碍物




# # Example usage:

# grid_map = np.array([[0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])

# point1 = [0, 1]
# # point1 = [1, 1]
# # point2 = [0, 1]
# point2 = [3, 2]


# if not no_obstacle_on_line(grid_map, point1, point2):
#     print("There is an obstacle on the line between the points.")
# else:
#     print("There is no obstacle on the line between the points.")
# # 可视化
# import matplotlib.pyplot as plt
# def plot_map_with_line(grid_map, point1, point2):
#     fig, ax = plt.subplots()
#     ax.imshow(grid_map, cmap='gray', origin='lower', extent=[0, grid_map.shape[1], 0, grid_map.shape[0]])

#     # 画线段
#     rr, cc = line(point1[0], point1[1], point2[0], point2[1])
#     ax.plot(cc, rr, color='red', linewidth=2)

#     # 标记起点和终点
#     ax.plot(point1[1], point1[0], marker='o', color='green', markersize=8, label='Start Point')
#     ax.plot(point2[1], point2[0], marker='o', color='blue', markersize=8, label='End Point')

#     ax.legend()
#     plt.show()
# plot_map_with_line(grid_map, point1, point2)