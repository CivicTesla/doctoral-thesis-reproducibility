import numpy as np
from AgentBS04 import Agent
from ComRate import ComPerMap

from Image2gridMap import create_grid_map,get_valid_from_whole_map

def EuclideanDistance(a,b): #point a and point b
    return np.linalg.norm(a-b)

def create_agents(N, ValidIdx,Goal):
    agents = []
    for i in range(N):
        # 从 ValidIdx 中随机选择一个有效位置
        random_idx = np.random.choice(len(ValidIdx))
        init_position = np.array(ValidIdx[random_idx])

        # 这里可以设置不同的初始速度，这里简单地使用随机值
        init_velocity = np.random.rand(2) * 5    # 生成随机的 x, y 速度

        init_dist = EuclideanDistance(init_position,Goal)


        agent = Agent(agent_id=i,init_position=init_position, init_velocity=init_velocity,init_dist=init_dist)
        agents.append(agent)

    return agents

# ##############################################################################################
# #Test Code
# import cv2
# def draw_agents_on_map(grid_map, agents):
#     map_with_agents = np.copy(grid_map)

#     for agent in agents:
#         position = tuple(map(int, agent.position))
#         cv2.circle(map_with_agents, position, radius=5, color=(255, 0, 0), thickness=-1)

#     return map_with_agents




# def main():
#     grid_map = create_grid_map('dilatedImg4orgMap.png')
#     valid_idx = get_valid_from_whole_map(grid_map)
#     gridMap = cv2.imread('dilatedImg4orgMap.png')

#     N = 20  # 设置代理数量
#     goal = np.array([300, 400])

#     # print('valid_idx',valid_idx)

#     # 创建 N 个代理，传入 valid_idx
#     agents = create_agents(N, valid_idx, goal)
#     for agent in agents:
#         y, x = agent.position
#         if grid_map[y, x] == 1:
#             print(f'agent ID {agent.ID} position is {agent.position}')
#             if tuple(agent.position) in valid_idx:
#                 print('yes!')



#     # 绘制代理在地图上的位置
#     map_with_agents = draw_agents_on_map(gridMap, agents)

#     # 显示地图
#     cv2.imshow('Map with Agents', map_with_agents)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
