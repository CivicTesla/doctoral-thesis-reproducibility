import numpy as np

from CollisionCheck import no_obstacle_on_line

def generate_grid_map(rows, cols):
    return np.zeros((rows, cols), dtype=int)


def is_point_inside_circle(point, center, radius):
    x, y = point
    cx, cy = center
    distance_squared = (x - cx)**2 + (y - cy)**2
    return distance_squared < radius**2


# def find_coordinates_inside_circle(center, radius, grid_map): #get valid points in circle
#     coordinates_inside_circle = []
#
#
#     for i in range(grid_map.shape[0]):
#         for j in range(grid_map.shape[1]):
#             # if is_point_inside_circle((i, j),center, radius):
#             #     pointsInCircle.append((i,j))
#             if is_point_inside_circle((i, j), center, radius) and grid_map[i, j] == 0:
#                 coordinates_inside_circle.append((i, j))
#
#     results_in_circle = []
#
#     for point in coordinates_inside_circle:
#         if no_obstacle_on_line(grid_map, center,point):
#             results_in_circle.append(point)
#     print('pointsInCircle',len(coordinates_inside_circle))
#     # Remove the center point if it exists in the results
#     results_in_circle = [point for point in results_in_circle if not np.array_equal(point, center)]
#     # results_in_circle = [point for point in coordinates_inside_circle if not np.array_equal(point, center)]
#     return results_in_circle

#DS 帮忙的高效圆圈代码
def find_coordinates_inside_circle(center, radius, grid_map):
    x, y = center
    x_min = max(0, x - radius)
    x_max = min(grid_map.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(grid_map.shape[1], y + radius + 1)

    coordinates_inside_circle = []
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if is_point_inside_circle((i, j), center, radius) and grid_map[i, j] == 0:
                coordinates_inside_circle.append((i, j))

    results_in_circle = []
    for point in coordinates_inside_circle:
        if no_obstacle_on_line(grid_map, center, point):
            results_in_circle.append(point)
    # print(f'LocalSensor56 pointsInCircle: {len(coordinates_inside_circle)}')

    # Remove the center point if it exists in the results
    results_in_circle = [point for point in results_in_circle if not np.array_equal(point, center)]
    # results_in_circle = [point for point in coordinates_inside_circle if not np.array_equal(point, center)]
    return results_in_circle

def find_coordinates_inside_ring(center, inner_radius, outer_radius, grid_map):
    coordinates_inside_ring = []

    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            distance_squared = (i - center[0])**2 + (j - center[1])**2
            if inner_radius**2 <= distance_squared <= outer_radius**2 and grid_map[i, j] == 0:
                coordinates_inside_ring.append((i, j))

    results_in_ring = []

    for point in coordinates_inside_ring:
        if no_obstacle_on_line(grid_map, center, point):
            results_in_ring.append(point)

    results_in_ring = [point for point in results_in_ring if not np.array_equal(point, center)]
    return results_in_ring



#Test Code ############################################################################


# # 示例：给定坐标 (4, 4) 为圆心，半径为 2
# center_coordinates = (111, 477)
# # center_coordinates = (400, 500)
# circle_radius = 15
#
# # 生成示例 gridMap
# from Image2gridMap import create_grid_map
# grid_map = create_grid_map('dilatedImg4orgMap.png')
#
# coordinates_inside_circle = find_coordinates_inside_circle(center_coordinates, circle_radius, grid_map)
# results_in_circle = []
# for point in coordinates_inside_circle:
#     if no_obstacle_on_line(grid_map, center_coordinates, point):
#         results_in_circle.append(point)
#     else:
#         print(f"路径 {center_coordinates} -> {point} 被障碍物阻挡")
# print('results_in_circle number: ',len(results_in_circle))
# print('grid_map',grid_map[111,477])
# # # grid_map = create_grid_map('dilatedImg.png')
# # print('grid_map[0,0]',grid_map[0,0])
# #
# #
# #
# #
# # # 寻找圆圈内包含0的坐标
# # result_coordinates = find_coordinates_inside_circle(center_coordinates, circle_radius, grid_map)
# # # print("Coordinates inside the circle:", result_coordinates)
# # print('result length',len(result_coordinates))
#
# # 画圆并显示
#
# import cv2
#
# def plot_circle(center, radius, grid_map):
#     # Create a copy of the grid map for drawing
#     map_image = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
#     map_image[grid_map == 1] = (0, 0, 0)
#     map_image[grid_map == 0] = (255, 255, 255)
#
#     # Draw the center point
#     cv2.circle(map_image, (center[1], center[0]), 5, (0, 255, 0), -1)
#
#     # Draw the circle
#     cv2.circle(map_image, (center[1], center[0]), radius, (0, 0, 255), 2)
#
#     # Show the image
#     cv2.imshow("Map with Circle", map_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# plot_circle(center_coordinates, circle_radius, grid_map)







##################################################################################
# from Image2gridMap import create_grid_map
# import cv2
# def draw_circle_on_map(grid_map, center, radius):
#     # Create a copy of the grid map for drawing
#     map_image = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
#     map_image[grid_map == 1] = (0, 0, 0)
#     map_image[grid_map == 0] = (255, 255, 255)
#
#     # Draw the center point
#     cv2.circle(map_image, (center[1],center[0]), 5, (0, 0, 255), -1)
#
#     # Draw the circle
#     cv2.circle(map_image, (center[1],center[0]), radius, (0, 0, 255), 2)
#
#     # Show the image
#     cv2.imshow("Map with Circle", map_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Get coordinates inside the circle
#     coordinates_inside_circle = []
#     for i in range(grid_map.shape[0]):
#         for j in range(grid_map.shape[1]):
#             if is_point_inside_circle((i, j), center, radius) and grid_map[i, j] == 0:
#                 coordinates_inside_circle.append((i, j))
#
#     return coordinates_inside_circle, map_image
#
#
# # Load the grid map
# grid_map = create_grid_map('dilatedImg4orgMap.png')
#
# # Set center and radius values for testing
# center = (241, 580)
# radius = 15
#
# # Call the draw function
# coordinates_inside_circle, modified_map = draw_circle_on_map(grid_map.copy(), center, radius)
#
# print(f'results number: {len(coordinates_inside_circle)}')
# print("Points inside the circle:", coordinates_inside_circle)
#
#
# # Display the modified map
# cv2.imshow('Modified Map with Circle', modified_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('Coordinates Inside Circle:', coordinates_inside_circle)