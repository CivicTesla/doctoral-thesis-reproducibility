from AgentP2P import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning
import cv2
from InitFunctionBS04 import create_agents
from Image2gridMap import create_grid_map,get_valid_from_whole_map
Epsilon = 5

def EuclideanDistance(a,b): #point a and point b
    return np.linalg.norm(a-b)

def RoundUpLinkSummarization(agents):
    UpLinkCount = 0
    UpLinkagents = []
    for agent in agents:
        UpLinkCount += 1
        if ComPerMap(agent.position):
            UpLinkagents.append(agent)
        else:
            agent.AConnection(None,-3)

    return UpLinkagents, UpLinkCount




def FindGbest(agents,Gbest,dist):
    # for agent in agents:
    #     current_dist = EuclideanDistance(agent.position, Goal)
    #     if current_dist < dist:
    #         dist = current_dist
    #         Gbest = agent.position
    if agents:
        bestAgent = min(agents, key=lambda agent: agent.dist)
        if bestAgent.dist == -2:
            return bestAgent.position,Epsilon #because now the dist is a flag
        if 0< bestAgent.dist < dist:
            return bestAgent.position,bestAgent.dist
    return  Gbest,dist
    




def CheckAchieved(agents):
    AchievedAgents = []
    for agent in agents:
        if agent.dist == -2:
            # print(f'agent ID ~~~: {agent.dist},agent position: {agent.position}')
            AchievedAgents.append(agent)
    return AchievedAgents









def RoundDownLink4PathPlanning(agent,Gbest):
	if ComPerMap(agent.position):
		agent.AConnection(Gbest,-1)
		# print('real one downlink')
	else: # tell agent randomly walk,because downlink failed
		agent.AConnection(Gbest,-3)


def RoundDownLinkSummarization(agents,Gbest,dist):
    DownLinkCount = 0
    # DownLinkagents = []
    for agent in agents:
        DownLinkCount += 1
        if ComPerMap(agent.position):
            agent.AConnection(Gbest,dist)
        else:
            agent.AConnection(None,-3)
    return DownLinkCount





    
def RoundDownLinkSingle(agent,Gbest,dist,N):
    DownCount = 0
    success = False
    for _ in range(max(1,N-1)):
        DownCount += 1
        if ComPerMap(agent.position):
            agent.AConnection(Gbest,dist)
            success = True
            break
    if not success:
        agent.AConnection(None,-3)
    return DownCount



def draw_agents_and_goal(gridMap, agents, Goal):
    map_copy = gridMap.copy()

    # 绘制地图
    map_image = np.zeros((map_copy.shape[0], map_copy.shape[1], 3), dtype=np.uint8)
    map_image[map_copy == 1] = (0, 0, 0)
    map_image[map_copy == 0] = (255, 255, 255)

    # 绘制代理和目标
    for agent in agents:
        cv2.circle(map_image, (agent.position[1], agent.position[0]), 5, (90, 30, 60), -1, cv2.LINE_AA)
    cv2.drawMarker(map_image, (Goal[1], Goal[0]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # 显示地图
    cv2.imshow('Map with Agents and Goal', map_image)

    # 检测按键
    key = cv2.waitKey(100)  # 每100毫秒检查一次按键

    # 如果按下Esc键，退出循环
    if key == 27:
        return False  # 返回 False 表示需要退出循环
    else:
        return True   # 返回 True 表示继续循环


#Main Code ###########################################################

def Single(N,ValidIdx,Goal):
    agents = create_agents(N,ValidIdx,Goal)
    # UpLinkNumberSum = 0
    # DownLinkNumberSum = 0
    UpLinkNumber = 0
    DownLinkNumber = 0
    AchievedAgents = []# to start 
    Uagents = []
    Gbest = None
    dist = np.Inf
    # LeftAgents = [a for a in agents if a not in AchievedAgents]
    LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]
    while(len(AchievedAgents) < 0.9 * N): #break loop means 90% agents get converged
        Uagents,UpLinkNum = RoundUpLinkSummarization(LeftAgents)
        UpLinkNumber += UpLinkNum
        AchievedAgents = CheckAchieved(Uagents)
        LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]
        
        Gbest,dist = FindGbest(Uagents,Gbest,dist)
        DownLinkNumber += RoundDownLinkSummarization(LeftAgents,Gbest,dist)


        
        # #draw
        # continue_loop = draw_agents_and_goal(display, agents, Goal)

        # # 在调用函数后检查是否需要继续循环
        # if not continue_loop:
        #     break  # 退出循环


    return UpLinkNumber,DownLinkNumber
            




    

#main Code
gridMap = create_grid_map('dilatedImg4orgMap.png')
display = create_grid_map('grid_map.png')
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

# AgentsNum = np.linspace(20,100,9).astype(int)
AgentsNum = range(10,100,10)
# AgentsNum = np.linspace(10,20,2).astype(int)

for aNum in AgentsNum:
    LoopUpLinkNumbers = []
    LoopDownLinkNumbers = []
    for i in range(15): #Here is the Monte Carlo Loop number
        print(f'Agents number {aNum}, Loop number {i}')
        UpLinkNumber,DownLinkNumber = Single(aNum,ValidIdx,Goal)
        # print('UpLinkNumber',UpLinkNumber)
        # print('DownLinkNumber',DownLinkNumber)
        LoopUpLinkNumbers.append(UpLinkNumber)
        LoopDownLinkNumbers.append(DownLinkNumber)


    np.save(f'Output/04BS/20-Agents{aNum}-LoopUpLinkNumber.npy',LoopUpLinkNumbers)
    np.save(f'Output/04BS/20-Agents{aNum}-LoopDownLinkNumber.npy',LoopDownLinkNumbers)












