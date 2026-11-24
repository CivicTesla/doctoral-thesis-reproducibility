import numpy as np

from CollisionCheck import no_obstacle_on_line

def generate_grid_map(rows, cols):
    return np.zeros((rows, cols), dtype=int)


def is_point_inside_circle(point, center, radius):
    x, y = point
    cx, cy = center
    distance_squared = (x - cx)**2 + (y - cy)**2
    return distance_squared < radius**2


def find_coordinates_inside_circle(center, radius, grid_map): #get valid points in circle
    coordinates_inside_circle = []
    pointsInCircle = []

    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            # if is_point_inside_circle((i, j),center, radius):
            #     pointsInCircle.append((i,j))
            if is_point_inside_circle((i, j), center, radius) and grid_map[i, j] == 0:
                coordinates_inside_circle.append((i, j))

    results_in_circle = []

    for point in coordinates_inside_circle:
        if no_obstacle_on_line(grid_map, center,point):
            results_in_circle.append(point)
    # print('pointsInCircle',len(coordinates_inside_circle))
    return results_in_circle




# # 示例：给定坐标 (4, 4) 为圆心，半径为 2
# center_coordinates = (107, 505)
# # center_coordinates = (400, 500)
# circle_radius = 30

# # 生成示例 gridMap
# from Image2gridMap import create_grid_map
# grid_map = create_grid_map('grid_map.png')
# # grid_map = create_grid_map('dilatedImg.png')
# print('grid_map[0,0]',grid_map[0,0])




# # 寻找圆圈内包含0的坐标
# result_coordinates = find_coordinates_inside_circle(center_coordinates, circle_radius, grid_map)
# # print("Coordinates inside the circle:", result_coordinates)
# print('result length',len(result_coordinates))

# # 画圆并显示
# import matplotlib.pyplot as plt
# def plot_circle(center, radius, grid_map):
#     fig, ax = plt.subplots()
#     ax.imshow(grid_map, cmap='gray', origin='lower', extent=[0, grid_map.shape[1], 0, grid_map.shape[0]])

#     # 以center为圆心，radius为半径画圆
#     circle = plt.Circle(center, radius, color='red', fill=False)
#     ax.add_patch(circle)

#     # 画绿色X表示圆心
#     ax.scatter(center[0], center[1], c='green', marker='x', s=100)

#     plt.show()
# plot_circle(center_coordinates, circle_radius, grid_map)







###################################################################################
# from Image2gridMap import create_grid_map
# import cv2
# def draw_circle_on_map(grid_map, center, radius):
#     # Create a copy of the grid map for drawing
#     map_image = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
#     map_image[grid_map == 1] = (0, 0, 0)
#     map_image[grid_map == 0] = (255, 255, 255)

#     # Draw the center point
#     cv2.circle(map_image, (center[1],center[0]), 5, (0, 0, 255), -1)

#     # Draw the circle
#     cv2.circle(map_image, (center[1],center[0]), radius, (0, 0, 255), 2)

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

#     return coordinates_inside_circle, map_image


# # Load the grid map
# grid_map = create_grid_map('dilatedImg4orgMap.png')

# # Set center and radius values for testing
# center = (570, 76)
# radius = 10

# # Call the draw function
# coordinates_inside_circle, modified_map = draw_circle_on_map(grid_map.copy(), center, radius)

# print(f'results number: {len(coordinates_inside_circle)}')
# print("Points inside the circle:", coordinates_inside_circle)


# # Display the modified map
# cv2.imshow('Modified Map with Circle', modified_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print('Coordinates Inside Circle:', coordinates_inside_circle)