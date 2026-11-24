# import numpy as np
# import math
# from skimage.draw import line


# def no_obstacle_on_line(grid_map, point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2

#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)

#     if x1 < x2:
#         sx = 1
#     else:
#         sx = -1

#     if y1 < y2:
#         sy = 1
#     else:
#         sy = -1

#     err = dx - dy
#     first_iteration = True

#     # Bresenham's Line Algorithm
#     while first_iteration or (x1 != x2 or y1 != y2):
#         first_iteration = False

#         if not (0 <= x1 < len(grid_map) and 0 <= y1 < len(grid_map[0])):
#             return False  # 超出地图范围，有障碍物

#         if grid_map[x1][y1] == 1:
#             return False  # 有障碍物

#         e2 = 2 * err

#         if e2 > -dy:
#             err = err - dy
#             x1 = x1 + sx

#         if e2 < dx:
#             err = err + dx
#             y1 = y1 + sy

#     # 检查终点
#     if grid_map[x2][y2] == 1:
#         return False  # 有障碍物

#     return True  # 没有障碍物





# # Example usage:
# import cv2
# from Image2gridMap import create_grid_map
# grid_map = create_grid_map('dilatedImg4orgMap.png')


# # point1 = np.array([[413,383]])
# # point2 = np.array([[300,400]]) #Goal point
# point1 = [413,383]
# point2 = [300,400]
# # point1 = [1, 1]
# # point2 = [0, 1]
# # point2 = [2, 2]


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

import cv2
import numpy as np
from skimage.draw import line

from Image2gridMap import create_grid_map

def _one_way_line_clear(grid_map, point1, point2):
    x1, y1 = map(int, point1)
    x2, y2 = map(int, point2)

    if not (0 <= x1 < grid_map.shape[0] and 0 <= y1 < grid_map.shape[1]):
        return False
    if not (0 <= x2 < grid_map.shape[0] and 0 <= y2 < grid_map.shape[1]):
        return False

    rr, cc = line(x1, y1, x2, y2)
    return not np.any(grid_map[rr, cc] == 1)


def no_obstacle_on_line(grid_map, point1, point2):
    return (
        _one_way_line_clear(grid_map, point1, point2)
        and _one_way_line_clear(grid_map, point2, point1)
    )

# # Test Code
# from skimage.draw import line
# grid_map = create_grid_map('dilatedImg4orgMap.png')

# point1 = [276, 251]
# point2 = [372, 237]#Goal point

# if not no_obstacle_on_line(grid_map, point1, point2):
#     print("There is an obstacle on the line between the points.")
# else:
#     print("There is no obstacle on the line between the points.")

# # Visualization using OpenCV
# def visualize_line_with_cv2(grid_map, point1, point2):
#     # Create a copy of the grid map for visualization
#     vis_map = cv2.cvtColor((grid_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#     # Draw the line on the visualization map
#     rr, cc = line(point1[0], point1[1], point2[0], point2[1])
#     vis_map[rr, cc] = [0, 0, 255]  # Red color for the line

#     # Mark the start and end points
#     cv2.circle(vis_map, (point1[1], point1[0]), 5, (0, 255, 0), -1)  # Green for start point
#     cv2.circle(vis_map, (point2[1], point2[0]), 5, (255, 0, 0), -1)  # Blue for end point

#     # Display the visualization map
#     cv2.imshow('Line Visualization', vis_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Call the visualization function
# visualize_line_with_cv2(grid_map, point1, point2)
