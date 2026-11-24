from AgentP2P import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning
import cv2
from InitFunctionP2P import create_agents
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




# def FindGbest(agents,Gbest,dist):
#     # for agent in agents:
#     #     current_dist = EuclideanDistance(agent.position, Goal)
#     #     if current_dist < dist:
#     #         dist = current_dist
#     #         Gbest = agent.position
#     if agents:
#         bestAgent = min(agents, key=lambda agent: agent.dist)
#         if bestAgent.dist == -2:
#             return bestAgent.position,Epsilon #because now the dist is a flag
#         if 0< bestAgent.dist < dist:
#             return bestAgent.position,bestAgent.dist
#     return  Gbest,dist
    
def FindGbest(agents,Gbest,dist):
    valid_agents = [agent for agent in agents if agent.dist>=0 or agent.dist == -2]

    if valid_agents:
        bestAgent = min(valid_agents, key=lambda agent: agent.dist)
        if bestAgent.dist == -2:
            return bestAgent.position, Epsilon  # because now the dist is a flag
        if 0 <= bestAgent.dist < dist:
            return bestAgent.position, bestAgent.dist
    else:
        print('Server58 法外之地: ')
    return Gbest,dist



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
    CovMoves = 0
    AchievedAgents = []# to start 
    i = 1
    # while(len(AchievedAgents) <= 0.9 * N): #break loop means 90% agents get converged
    while(len(AchievedAgents) < 1): #break loop means 90% agents get converged
        for agent in agents:
            other_agents = [a for a in agents if a != agent] 
            for Oagent in other_agents:
                UpLinkNumber +=1 # no matter success or not ,uplink counts for agent
                if ComPerMap(agent.position) and ComPerMap(Oagent.position): #Uplink for agent and Downlink for Oagent
                    DownLinkNumber += 1 #only uplink agent succeed, downlink Oagent counts
                    if agent.dist < EuclideanDistance(Oagent.Gbest,Goal):
                        Oagent.Gbest = agent.position
                        if Oagent.getFlag == False:
                            Oagent.getFlag = True

        for agent in agents:
            if agent.getFlag == False:
                agent.AConnection(-3)  # means no agents get contacted, so tell agent to randomly walk

            elif agent.getFlag == True:
                agent.AConnection(1)
                agent.getFlag = False  # wait for next round

    
        
        AchievedAgents = CheckAchieved(agents)
        RoundMoves = len(agents)-len(AchievedAgents)
        CovMoves += RoundMoves

        # print('Round number: ',i)
        # print('CovMoves: ',CovMoves)
        i += 1

        #draw
        # continue_loop = draw_agents_and_goal(display, agents, Goal)

        # #在调用函数后检查是否需要继续循环
        # if not continue_loop:
        #    break  # 退出循环
        # print(f'Uplink num {UpLinkNumber}, DownLinkNum {DownLinkNumber}')

    return UpLinkNumber,DownLinkNumber,i,CovMoves
            




    

#main Code
gridMap = create_grid_map('dilatedImg4orgMap.png')
display = create_grid_map('grid_map.png')
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

# AgentsNum = np.linspace(20,100,9).astype(int)
AgentsNum = range(100,110,10)
# AgentsNum = np.linspace(10,20,2).astype(int)

for aNum in AgentsNum:
    LoopUpLinkNumbers = []
    LoopDownLinkNumbers = []
    Loopi = []
    LoopCovMoves = []
    for i in range(10): #Here is the Monte Carlo Loop number
        print(f'Agents number {aNum}, Loop number {i}')
        UpLinkNumber,DownLinkNumber,Round,CovMoves = Single(aNum,ValidIdx,Goal)
        # print('UpLinkNumber',UpLinkNumber)
        # print('DownLinkNumber',DownLinkNumber)
        LoopUpLinkNumbers.append(UpLinkNumber)
        LoopDownLinkNumbers.append(DownLinkNumber)
        Loopi.append(Round)
        LoopCovMoves.append(CovMoves)



    np.save(f'Output/03P2P/04-Agents{aNum}-LoopUpLinkNumber.npy',LoopUpLinkNumbers)
    np.save(f'Output/03P2P/04-Agents{aNum}-LoopDownLinkNumber.npy',LoopDownLinkNumbers)
    np.save(f'Output/03P2P/04-Agents{aNum}-Loopi.npy',Loopi)
    np.save(f'Output/03P2P/04-Agents{aNum}-LoopCovMoves.npy',LoopCovMoves)











