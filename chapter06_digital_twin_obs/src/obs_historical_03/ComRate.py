import numpy as np
import random

# 用你的.npy文件路径替换下面的文件路径
file_path = 'baked_per_map_600x600_1_300_p15.npy'

# 使用NumPy的load函数加载.npy文件
perMap = np.load(file_path)

# print('data',data)

def ComPerMap(position): #看看位置上有没有联通,没联通就返回-1,表示没信号,有就该咋就咋
    row, col = map(int, position)
    if not (0 <= row < perMap.shape[0] and 0 <= col < perMap.shape[1]):
        return False
    return random.randint(1, 1000) > perMap[row, col]
     

def ComLogic(agent,Gbest,dist):
    if ComPerMap(agent.position):
        return Gbest,dist
    else:
        dist = -3
    return Gbest,dist



# # Test program for ComLogic#############
# from Image2gridMap import create_grid_map,get_valid_from_whole_map
# import Agent
# def create_agents(N, ValidIdx):
#     agents = []
#     for _ in range(N):
#         # 从 ValidIdx 中随机选择一个有效位置
#         random_idx = np.random.choice(len(ValidIdx))
#         init_position = ValidIdx[random_idx]

#         # 这里可以设置不同的初始速度，这里简单地使用随机值
#         init_velocity = np.random.rand(2) * 5    # 生成随机的 x, y 速度

#         agent = Agent.Agent(init_position=init_position, init_velocity=init_velocity)
#         agents.append(agent)

#     return agents

# if __name__ == "__main__":
#     gridMap = create_grid_map('org_map.png')

#     ValidIdx = get_valid_from_whole_map(gridMap)

#     N = 5  # 设置agent数量

#     # 创建 N 个代理，传入 ValidIdx
#     agents = create_agents(N, ValidIdx)

#     Gbest = np.array([5, 6])
#     dist = 10
#     for agent in agents:
#         Gb,di = ComLogic(agent,Gbest,dist)
#         print('Gb,di',Gb,di)
