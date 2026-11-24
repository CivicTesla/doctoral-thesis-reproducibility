# import copy
# import math
import random
# import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np

from Image2gridMap import create_grid_map, get_valid_coordinates, get_valid_from_whole_map

from CircleArea import get_Endpoint, extract_circle_region

from CollisionCheck import no_obstacle_on_line



# gridMap = create_grid_map('grid_map.png')
# gridMap = create_grid_map('org_map.png')
gridMap = create_grid_map('dilatedImg4orgMap.png')

ValidIdx = get_valid_from_whole_map(gridMap)




# def sample(gridMap, point1, point2):
#     # 获取圆内的有效区域
#     circleRegion = extract_circle_region(gridMap, point1, point2)

#     # 获取有效区域的坐标
#     validRegion = get_valid_coordinates(circleRegion, gridMap)

#     # 如果可选样本不足 10 个，直接返回全部可用样本
#     if len(validRegion) < 30:
#         return validRegion

#     # 从有效区域坐标中随机选择 10 个点
#     random_samples = random.sample(validRegion, 30)

#     return random_samples


def sample(gridMap, point1, point2):
    # # 获取圆内的有效区域
    # circleRegion = extract_circle_region(gridMap, point1, point2)

    # # 获取有效区域的坐标
    # validRegion = get_valid_coordinates(circleRegion, gridMap)

    # validRegion = ValidIdx



    # # 随机选择策略索引，0表示策略1，1表示策略2
    # selected_strategy = np.random.choice([0, 1])

    

    # # 根据选择的策略赋值给 validRegion
    # if selected_strategy == 0:
    #     # 获取圆内的有效区域
    #     circleRegion = extract_circle_region(gridMap, point1, point2)

    #     # 获取有效区域的坐标
    #     validRegion = get_valid_coordinates(circleRegion, gridMap)
    # else:
    #     validRegion = ValidIdx

    validRegion = ValidIdx




    

    # 如果可选样本不足 10 个，直接返回全部可用样本
    if len(validRegion) < 10:
        return validRegion

    # 计算每个子区域的大小
    num_samples = len(validRegion)
    samples_per_region = num_samples // 50

    # 从每个子区域中各选择一个点
    random_samples = []
    for i in range(50):
        start_index = i * samples_per_region
        end_index = (i + 1) * samples_per_region
        region_samples = validRegion[start_index:end_index]
        if region_samples:
            random_samples.append(random.choice(region_samples))

    return random_samples



class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dist = 0.0
        self.parent = None

# def find_path(gridMap, point1, point2, samples):
#     def distance(node1, node2):
#         return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5

#     nodes = [Node(x, y) for x, y in samples]
#     node1 = Node(point1[0], point1[1])
#     node2 = Node(point2[0], point2[1])
    
#     open_set = [node1]
#     visited = set()
    
#     while open_set:
#         current_node = open_set.pop(0)
#         visited.add(current_node)
        
#         if current_node == node2:
#             # Reconstruct the path
#             path = [current_node]
#             while current_node.parent:
#                 path.append(current_node.parent)
#                 current_node = current_node.parent
#             path.reverse()
#             return path
        
#         neighbors = [node for node in nodes if node != current_node and no_obstacle_on_line(gridMap, (current_node.x, current_node.y), (node.x, node.y))]
        
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 if current_node.dist + distance(current_node, neighbor) < neighbor.dist or neighbor not in open_set:
#                     neighbor.dist = current_node.dist + distance(current_node, neighbor)
#                     neighbor.parent = current_node
#                     open_set.append(neighbor)
    
#     return None  # No path found

# def find_path(gridMap, point1, point2, samples):
#     def distance(node1, node2):
#         return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5

#     nodes = [Node(x, y) for x, y in samples]
#     node1 = Node(point1[0], point1[1])
#     node2 = Node(point2[0], point2[1])
    
#     open_set = [node1]
#     visited = set()

#     # 1. 设置p1_child的parent和dist
#     for node in nodes:
#         if no_obstacle_on_line(gridMap, (node.x, node.y), (node1.x, node1.y)):
#             node.parent = node1
#             node.dist = distance(node, node1)

#     # 2. 查找p1_child点中能够连接到p2的点
#     p2_candidates = [node for node in nodes if node.parent == node1 and no_obstacle_on_line(gridMap, (node.x, node.y), (node2.x, node2.y))]
#     if p2_candidates:
#         # 选择距离p2最近的点，将其设置为p2的parent
#         closest_p2_candidate = min(p2_candidates, key=lambda n: distance(n, node2))
#         closest_p2_candidate.parent = node2
#         open_set.append(closest_p2_candidate)

#     while open_set:
#         current_node = open_set.pop(0)
#         visited.add(current_node)

#         if current_node == node2:
#             # Reconstruct the path
#             path = [current_node]
#             while current_node.parent:
#                 path.append(current_node.parent)
#                 current_node = current_node.parent
#             path.reverse()
#             return path

#         # 3. 如果没有p2_candidates，尝试找到没有parent的点，并与离自己最近的有parent的点相连
#         if not p2_candidates:
#             for node in nodes:
#                 if node.parent is None:
#                     closest_with_parent = min(nodes, key=lambda n: distance(node, n) if n.parent else float('inf'))
#                     node.parent = closest_with_parent
#                     node.dist = distance(node, closest_with_parent)
#                     # Check if this newly connected node can reach p2
#                     if no_obstacle_on_line(gridMap, (node.x, node.y), (node2.x, node2.y)):
#                         p2_candidates.append(node)
#                         closest_p2_candidate = min(p2_candidates, key=lambda n: distance(n, node2))
#                         closest_p2_candidate.parent = node2
#                         open_set.append(closest_p2_candidate)

#     return None  # No path found

def find_path(gridMap, point1, point2, samples):
    def distance(node1, node2):
        return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5

    nodes = [Node(x, y) for x, y in samples]
    node1 = Node(point1[0], point1[1])
    node2 = Node(point2[0], point2[1])
    
    # 1.检查p1和p2能不能直接相连
    if no_obstacle_on_line(gridMap, (node2.x, node2.y), (node1.x, node1.y)):
        # Reconstruct the path
        path = [node1, node2]
        return path

    # 2. 设置p1_child的parent和dist
    for node in nodes:
        if no_obstacle_on_line(gridMap, (node.x, node.y), (node1.x, node1.y)):
            node.parent = node1
            node.dist = distance(node, node1)

    # 3. 查找有parent的点能够连接到p2的点
    p2_candidates = [node for node in nodes if node.parent and no_obstacle_on_line(gridMap, (node.x, node.y), (node2.x, node2.y))]
    if p2_candidates:
        # for p2c in p2_candidates:
        #     print('p2_candidates',p2c.x,p2c.y)
        # 选择距离p2最近的点，将其设置为p2的parent
        closest_p2_candidate = min(p2_candidates, key=lambda n: distance(n, node2))
        node2.parent = closest_p2_candidate

        # Reconstruct the path
        path = [node2]
        current_node = node2
        while current_node.parent:
            path.append(current_node.parent)
            current_node = current_node.parent
        path.reverse()
        return path

    # # 4. 遍历没有parent的node，每个没parent的node选择离自己最近的有parent的node作为parent
    # for node in nodes:
    #     if node.parent is None:
    #         closest_with_parent = min(nodes, key=lambda n: distance(node, n) if n.parent else float('inf'))
    #         node.parent = closest_with_parent
    #         node.dist = distance(node, closest_with_parent)

    #         # Check if this newly connected node can reach p2
    #         if no_obstacle_on_line(gridMap, (node.x, node.y), (node2.x, node2.y)):
    #             # Reconstruct the path
    #             path = [node2]
    #             current_node = node2
    #             while current_node.parent:
    #                 path.append(current_node.parent)
    #                 current_node = current_node.parent
    #             path.reverse()
    #             return path

    # # 4. 如果没有从p2能回溯到p1的path，返回None
    # return None


    # # 4. 遍历有parent但没child的点s1，找出离它们最近的点s2，将s2的parent设置为s1
    # for node1 in nodes:
    #     if node1.parent and not any(node2.parent == node1 for node2 in nodes):
    #         closest_childless = min(nodes, key=lambda n: distance(node1, n) if not n.parent else float('inf'))
    #         closest_childless.parent = node1
    #         closest_childless.dist = distance(closest_childless, node1)

    #         # Check if this newly connected node can reach p2
    #         if no_obstacle_on_line(gridMap, (closest_childless.x, closest_childless.y), (node2.x, node2.y)):
    #             # Reconstruct the path
    #             path = [node2]
    #             current_node = node2
    #             while current_node.parent:
    #                 path.append(current_node.parent)
    #                 current_node = current_node.parent
    #             path.reverse()
    #             return path

    # # 5. 如果没有从p2能回溯到p1的path，返回None
    # return None

    # 4. 遍历有parent但没child的点s1，找出离它们最近的，且没有障碍物的点s2，将s2的parent设置为s1
    for node1 in nodes:
        if node1.parent and not any(node2.parent == node1 for node2 in nodes):
            # 找到离node1最近，且没有障碍物的点s2
            candidate_nodes = [
                node for node in nodes if node != node1 and node.parent is None and no_obstacle_on_line(gridMap, (node1.x, node1.y), (node.x, node.y))
            ]
            
            if candidate_nodes:
                # 如果s2有多个s1，选择距离s2最近的s1作为parent
                closest_childless = min(candidate_nodes, key=lambda n: distance(node1, n))
                closest_childless.parent = node1
                closest_childless.dist = distance(closest_childless, node1)



    # 5. 查找有parent的点能够连接到p2的点
    p2_candidates = [node for node in nodes if node.parent and no_obstacle_on_line(gridMap, (node.x, node.y), (node2.x, node2.y))]

    if p2_candidates:
        # 选择距离p2最近的点，将其设置为p2的parent
        closest_p2_candidate = min(p2_candidates, key=lambda n: distance(n, node2))
        node2.parent = closest_p2_candidate

        # Reconstruct the path
        path = [node2]
        current_node = node2
        while current_node.parent:
            path.append(current_node.parent)
            current_node = current_node.parent
        path.reverse()
        return path

    # 6. 如果没有从p2能回溯到p1的path，返回None
    return None




# #测试用Code
# # 调用 sample 函数
# samples = sample(gridMap, point1, point2)

# # 打印随机选择的 个点
# print('随机选择的多个点：', samples)



# # 检查前两个样本点是否有障碍物
# if not no_obstacle_on_line(gridMap, samples[0], samples[1]):
#     print("前两个样本没有障碍物")
# else:
#     print("前两个样本有障碍物")



#测试find_path()
# 查找路径
# path = find_path(gridMap, point1, point2, samples)






# def draw_path_with_samples(gridMap, point1, point2, samples, path):
def draw_path_with_samples(gridMap, point1, point2, path):
    map_copy = gridMap.copy()

    # 绘制地图
    map_image = np.zeros((map_copy.shape[0], map_copy.shape[1], 3), dtype=np.uint8)
    map_image[map_copy == 1] = (0, 0, 0)
    map_image[map_copy == 0] = (255, 255, 255)

    # 绘制起点和终点
    cv2.circle(map_image, (point1[1], point1[0]), 10, (0, 0, 255), -1)
    cv2.circle(map_image, (point2[1], point2[0]), 10, (0, 255, 0), -1)

    # # 绘制随机选择的点
    # for sample in samples:
    #     cv2.circle(map_image, (sample[1], sample[0]), 3, (255, 0, 0), -1)

    # # 绘制大圈
    # center_x, center_y, radius = get_Endpoint(point1, point2)
    # cv2.circle(map_image, (center_y, center_x), radius, (0, 0, 0), 2)

    # if path:
    #     # 绘制路径
    #     for i in range(len(path) - 1):
    #         node1 = path[i]
    #         node2 = path[i + 1]
    #         cv2.line(map_image, (int(node1.y), int(node1.x)), (int(node2.y), int(node2.x), (0, 255, 0), 2))

    if path:
        # 绘制路径
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            cv2.line(map_image, (int(node1.y), int(node1.x)), (int(node2.y), int(node2.x)), (0, 255, 0), 2)


    # 显示地图
    cv2.imshow("Map with Samples and Path", map_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def SesamePathPlanning(gridMap,point1,point2):
    # 查找路径
    for i in range(1000): # 会计划运行Pathplanning 50次,一旦找到路径就返回,实际可能不到50次
        # print(f'the {i}th time')
        samples = sample(gridMap, point1, point2)
        path = find_path(gridMap, point1, point2, samples)
        if path == None:
            continue
        else:
            break
    return path



# # # # # TestCode ##############################

# import cv2
# point1 = [372, 237] #例子start point
# # point1 = [300,400]
# # # point1 = [426,79]
# point2 = [276, 251] #例子goal point


# # # random_idx = np.random.choice(len(ValidIdx))
# # # point1 = np.array(ValidIdx[random_idx])
# # # random_idx = np.random.choice(len(ValidIdx))
# # # point2 = np.array(ValidIdx[random_idx])


# # # samples = sample(gridMap, point1, point2)
# path = SesamePathPlanning(gridMap,point1,point2)
# path1 = path[1:]


# path_coordinates = [(node.x, node.y) for node in path]
# print(f"Path: {path_coordinates}")


# path_coordinates1 = [(node.x, node.y) for node in path1]
# print(f"Path: {path_coordinates1}")
# print(f"Path without first element: {path[0].x,path[0].y}")

# # for node in path:
# #     print(f"Node: ({node.x}, {node.y})")
# # #     print('path next point',np.array([path[1].x,path[1].y]))
# # # print('path type',type(path))


# # # 绘制地图和路径
# # # draw_path_with_samples(gridMap, point1, point2, samples, path)
# draw_path_with_samples(gridMap, point1, point2, path)



