import numpy as np

# def polarToCartesian(velocity_magnitude, velocity_angle):
#     # 将极坐标转换为直角坐标
#     velocity_x = velocity_magnitude * np.cos(velocity_angle)
#     velocity_y = velocity_magnitude * np.sin(velocity_angle)#此处可能需要加一个负号

#     return velocity_x, velocity_y



# import numpy as np

# def histogramFun(center, radius, GridMap):
#     # 获取 GridMap 的形状
#     rows, cols = GridMap.shape

#     # 生成网格坐标
#     x, y = np.meshgrid(np.arange(cols), np.arange(rows))

#     # 创建掩码，将值为0的格子排除在计算之外
#     mask = GridMap == 1

#     # 计算每个点到圆心的距离，只考虑值为1的格子
#     distances = np.sqrt((x[mask] - center[0])**2 + (y[mask] - center[1])**2)

#     # 将距离小于等于半径的点标记为 1，否则标记为 0
#     circle_mask = distances <= radius

#     # 获取圆内的坐标
#     circle_coords = np.column_stack((x[mask][circle_mask], y[mask][circle_mask]))

#     # 计算每个点相对于圆心的极坐标角度
#     angles = np.arctan2(circle_coords[:, 1] - center[1], circle_coords[:, 0] - center[0])

#     # 将角度映射到 [0, 2*pi) 范围
#     angles = (angles + 2 * np.pi) % (2 * np.pi)

#     # 将角度分为 8 个扇区
#     sector_indices = np.digitize(angles, np.linspace(0, 2*np.pi, 9)) - 1

#     # 计算每个点到圆心的归一化距离
#     normalized_distances = distances[circle_mask] / radius
#     print('normalized_distances: \n', normalized_distances)
#     print('normalized_distances size: ', normalized_distances.shape)

#     # 计算 h 值，alpha合理范围在[1,2]
#     alpha = 2
#     h_values = (1 - normalized_distances) ** alpha

#     # 根据 sector_indices 统计每个扇区的和
#     sector_sums = np.zeros(8)
#     for i in range(8):
#         sector_sums[i] = np.sum(h_values[sector_indices == i])

#     # 计算每个扇区的中点角度，这里使用每个扇区的起始角度
#     sector_start_angles = np.linspace(0, 2 * np.pi, 9)[:-1]
#     sector_midpoints = (sector_start_angles + np.roll(sector_start_angles, -1)) / 2

#     return sector_sums, sector_midpoints



# # Test Code ########################################################
# # 生成一个 10x10 的 GridMap
# # np.random.seed(42)
# # GridMap = np.random.choice([0, 1], size=(10, 10), p=[0.5, 0.5])
# # print('GridMap\n',GridMap)
# GridMap = np.array([
#     [0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
#     [1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
#     [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 1, 0, 1, 0, 0, 0]
# ])

# print('sum GridMap',np.sum(GridMap))

# center = (4, 5)
# radius = 5

# result, sector_ranges = histogramFun(center, radius, GridMap)
# print("Sector Sums with h = (1-ri/rmax)^alpha: \n", result)
# print("Sector Midpointss:\n", sector_ranges)

# #弧度变角度
# radians = 0.39
# degrees = np.degrees(radians) #0.39/3.14*180=22.35

# print(degrees)

def calculate_direction(robot_position, target_position):
    # Calculate the vector from robot to target
    vector_to_target = np.array(target_position) - np.array(robot_position)

    # Calculate the angle in radians
    angle_radians = np.arctan2(vector_to_target[1], vector_to_target[0])

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # Ensure the angle is between 0 and 360 degrees
    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees

# Example usage
robot_position = (1, 1)
target_position = (4, 3)

target_direction = calculate_direction(robot_position, target_position)
print(f"Target Direction: {target_direction} degrees")