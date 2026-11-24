import numpy as np

def vfh(center, radius, gridMap):
    num_sectors = 12
    alpha = 2

    angles = np.arange(-45, 315, 360/num_sectors) #-45度到315度
    # print('angles',angles)
    sector_middle = [(start + end) / 2 for start, end in zip(angles, np.roll(angles, -1))]
    
    obstacle_density = np.zeros(num_sectors)

    for i in range(num_sectors):
        sector_start, sector_end = angles[i], angles[(i+1) % num_sectors]
        sector_points = find_points_in_sector(center, radius, gridMap, sector_start, sector_end)
        # print(f'sector_points {i}: \n',sector_points)
        
        for point in sector_points:
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            weighted_range = (1.0 - distance/radius)**alpha
            obstacle_density[i] += weighted_range

    h_max = 0.64
    sector_occupied = np.zeros(num_sectors, dtype=bool)
    sector_occupied[obstacle_density > h_max] = True
    
    candidate_sectors = np.where(~sector_occupied)[0]
    candidate_middle_values = [sector_middle[i] for i in candidate_sectors]
    
    return candidate_middle_values



def find_points_in_sector(center, radius, gridMap, start_angle, end_angle):
    points_in_sector = []

    # 遍历整个圆圈
    for i in range(gridMap.shape[0]):
        for j in range(gridMap.shape[1]):
            # 检查网格上是否为1
            if gridMap[i, j] == 1:
                # 计算点与圆心的夹角
                angle = calculate_angle(center, (i, j))

                # 检查点是否在扇区范围内
                if start_angle <= angle <= end_angle:
                    points_in_sector.append((i, j))

    return points_in_sector

def calculate_angle(center, point):
    x, y = point[0] - center[0], point[1] - center[1]
    angle = np.degrees(np.arctan2(y, x))-90 #因为一个几何位置与xy轴的原因所以这样
    angle = angle%360 #统一了-135度与225度
    if angle>=315: #归类sector0
        angle -=360
    return angle


# # TestCode######################################################################
# gridMap = np.array([
#     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1], #0
#     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],#1
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],#2
#     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],#3
#     [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],#4
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],#5
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],#6
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],#7
#     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],#8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#9
#     #0  1  2  3  4  5  6  7  8  9
# ])
# """
# 0,3 0,7 s6 0,8 0,9
# 1,5 1,7 s6
# 2,2
# 3,3 3,5 s6
# 4,5 4,6 4,8 4,9 s4
# 5,2
# 6,3 s0
# 7,7 s3 7,9 
# 8,4 s1 8,7 s2
# """
# center = (4, 4)
# radius = 5

# # candidate_sectors, sectors_middle = vfh(center, radius, gridMap)
# sectors_middle = vfh(center, radius, gridMap)
# # print("候选扇区:", candidate_sectors)
# print("候选扇区对应的中间值:", sectors_middle)

# sectorNullangle = calculate_angle(center,(6,3))
# print('sector 0',sectorNullangle)