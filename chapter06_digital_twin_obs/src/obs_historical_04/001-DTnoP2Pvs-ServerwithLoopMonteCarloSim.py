import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning
import cv2
from InitFunction import create_agents
from Image2gridMap import create_grid_map,get_valid_from_whole_map
Epsilon = 5

def find_map_center(grid_map):
    num_rows = len(grid_map)
    num_cols = len(grid_map[0])

    center_row = num_rows // 2
    center_col = num_cols // 2

    return center_row, center_col

def EuclideanDistance(a,b): #point a and point b
    return np.linalg.norm(a-b)

def RoundUpLinkSummarization(agents):
	# UpLinkCount = 0
	UpLinkagents = []
	for agent in agents:
		# UpLinkCount += 1 # no matter successful or fail, uplink tried once
		if ComPerMap(agent.position):
			UpLinkagents.append(agent)
		else: # tell agent randomly walk
		    agent.AConnection(None,-3)
	return UpLinkagents


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
    
# def FindGbest(agents,Gbest,dist):
#     valid_agents = [agent for agent in agents if agent.dist>=0 or agent.dist == -2]

#     if valid_agents:
#         bestAgent = min(valid_agents, key=lambda agent: agent.dist)
#         if bestAgent.dist == -2:
#             return bestAgent.position, Epsilon  # because now the dist is a flag
#         if 0 <= bestAgent.dist < dist:
#             return bestAgent.position, bestAgent.dist
#     else:
#         print('Server58 法外之地: ')
#     return Gbest,dist

def ClassifyPathPSO(agents):
    # i = 0
    PSOagents = []
    PathAgents = []
    for agent in agents:
        if agent.dist >= 2: #save some space for achieved agents
            PSOagents.append(agent) 
        if agent.dist == -1:
            PathAgents.append(agent)
        # if Udist == -2 and agent.ID not in AchievedAgents:
        #    AchievedAgents.append(agent.ID)
        # 检查 AchievedAgents 是否已定义

        # 检查 Udist 是否等于 -2
        # if agent.ID not in AchievedAgents:
        #     if Udist == -2:
        #     # 检查 agent.ID 是否在 AchievedAgents 中
            
                # AchievedAgents.append(agent.ID)
    return PSOagents,PathAgents

def CheckAchieved(agents):
    AchievedAgents = []
    for agent in agents:
        if agent.dist == -2:
            # print(f'agent ID ~~~: {agent.dist},agent position: {agent.position}')
            AchievedAgents.append(agent)
    return AchievedAgents




def PathAssignments(agents,Gbest):# for agent.dist = -1
    PathTable = []
    for agent in agents:
        path = SesamePathPlanning(gridMap, agent.position.tolist(), Gbest.tolist())
        if path is None:
            print('path == None !!!!!!!!!')
            print(f'agent ID {agent.ID} position is {agent.position}')
            print(f'Gbest is {Gbest}')
            continue

        path = path[1:]  # the first element is agent's position,so remove it
        # if len(path) >= 3:
        #     path = path[:-2]  # To let agent get so PSO experience

        PathTable.append((agent.ID, path))
    return PathTable


def RoundDownLink4PathPlanning(agent,Gbest):
	if ComPerMap(agent.position):
			agent.AConnection(Gbest,-1)
			# print('real one downlink')
	else: # tell agent randomly walk,because downlink failed
		    agent.AConnection(Gbest,-3)


def RoundDownLinkSummarization(agents,Gbest,dist):
    DownLinkCount = 0
    DownLinkagents = []
    for agent in agents:
        # DownLinkCount += 1 # no matter successful or fail, uplink tried once
        if ComPerMap(agent.position):
            agent.AConnection(Gbest,dist)
            # print('real one downlink')
        else: # tell agent randomly walk,because downlink failed
            agent.AConnection(None,-3)
    return DownLinkCount
    
def RoundDownLinkSingle(agent,Gbest,dist):
	if ComPerMap(agent.position):
		agent.AConnection(Gbest,dist)
	else:
		agent.AConnection(None,-3)

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
    agents = create_agents(N, ValidIdx,Goal)

    UpLinkNumber = len(agents)#for the first up link summary

    Uagents = RoundUpLinkSummarization(agents) #first uplink agent's position and dist

    DownLinkNumber = len(Uagents) #first downlink number

    # print('number of initial uplink agents',len(Uagents))

    Gbest,dist = FindGbest(Uagents,None,np.Inf) # initial FindGbest
    # dist = -1
    CovMoves = N #here to set a initial sum of RoundMoves

    HistoryGbest,Historydist = Gbest,dist

    DownLinkCount = RoundDownLinkSummarization(Uagents,Gbest,dist)

    # print('Initial DownLinkCount',DownLinkCount)

    # print('initial Gbest and dist',HistoryGbest,Historydist)

    #First Loop Uplink#######################################################
    Uagents = RoundUpLinkSummarization(agents)#uplink successfl
    # best_agent = FindGbest(Uagents) #PSO Gbest and dist

    # print('number of first loop uplink agents',len(Uagents))
    AchievedAgents = CheckAchieved(Uagents)
    UagentsA = [agent for agent in Uagents if agent not in AchievedAgents]#uplink agents wating for assignments
    DownLinkNumber += len(UagentsA)
    UpLinkNumber += len(agents)
    # print(f'UpLinkNumber: {UpLinkNumber}, DownLinkNumber: {DownLinkNumber}~~~~~~~~~~~~~~~~~~~~~')

    PSOagents,PathAgents = ClassifyPathPSO(UagentsA)
    # print('PathAgents number',len(PathAgents))
    # print('PSOagents number',len(PSOagents))
    # print('AchievedAgents number',len(AchievedAgents))
    # PSOagents1,PathAgents1,AchievedAgents1,PathKeepAgents1 = ClassifyNew(Uagents)
    # print('PathKeepAgents number',len(PathKeepAgents))
    # for agent in AchievedAgents:
    #   print('AchievedAgent ID: ', agent.ID)

    #keep Gbest and dist are the best
    if AchievedAgents:
        Gbest,dist = AchievedAgents[0].position,AchievedAgents[0].dist
    else:
        Gbest,dist = FindGbest(PSOagents,HistoryGbest,Historydist)
        

    #first Downlink############################################################
    #for PSO downlink
    PSODownCount = RoundDownLinkSummarization(PSOagents,Gbest,dist)
    PathCount = 0

    PathTable = PathAssignments(PathAgents,Gbest)


    for agent in PathAgents:
        for agent_id, path in PathTable:
            if agent.ID == agent_id:
                GbestP = np.array([path[0].x,path[0].y]) if path[0] is not None else Gbest
                distP = -1 if path is not None else -3 #-3 means no path, so tell Agent randomly walk
                RoundDownLinkSingle(agent, GbestP, distP)
                PathCount += 1


    # print('PSOCount: ',PSODownCount)
    # print('PathCount: ',PathCount)
    # DownLinkCount = PathCount + PSODownCount
    # print('DownLinkCount: ',DownLinkCount)

    if AchievedAgents:
        AchievedNumber = len(AchievedAgents)
    else:
        AchievedNumber = 0

    LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]
    i = 0
    #Big Loop Start#######################################
    while(True):
        # print(f'the {i+1}th time for the loop!!!!!!!!!!!!!!')
        i += 1
        Uagents = RoundUpLinkSummarization(LeftAgents) 
        # print('number of Big loop uplink agents',len(Uagents))
        # if not PathTableOld :
        #     PathTableOld = []#here because for the beginning there's no PathTable
            # print('PathOld is None~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        AchievedAgentsU = CheckAchieved(Uagents)
        UagentsA = [agent for agent in Uagents if agent not in AchievedAgentsU]#uplink agents wating for assignments
        
        # 将 AchievedAgentsU 中的代理合并到 AchievedAgents 中
        if AchievedAgents is None:
            AchievedAgents = AchievedAgentsU
        else:
            AchievedAgents.extend(AchievedAgentsU)

        PSOagents,PathAgents = ClassifyPathPSO(UagentsA)
        # print('PathAgents number',len(PathAgents))
        # print('PSOagents number',len(PSOagents))
        # print('AchievedAgents number',len(AchievedAgents))

        #keep Gbest and dist are the best
        if AchievedAgents:
            Gbest,dist = AchievedAgents[0].position,AchievedAgents[0].dist
        else:
            Gbest,dist = FindGbest(PSOagents,HistoryGbest,Historydist)

        if dist < Historydist:
            HistoryGbest,Historydist = Gbest,dist
        Gbest, dist = HistoryGbest, Historydist 

        
        UpLinkNumber += len(LeftAgents)

        DownLinkNumber += len(UagentsA)
        # print(f'UpLinkNumber: {UpLinkNumber}, DownLinkNumber: {DownLinkNumber}~~~~~~~~~~~~~~~~~')

        #Downlink ####################################################################
        #for PSO downlink##############
        PSODownCount = RoundDownLinkSummarization(PSOagents,HistoryGbest,Historydist)
        PathCount = 0

        # PathAgents downlink##########
        PathTable = PathAssignments(PathAgents,Gbest)

        for agent in PathAgents:
            for agent_id, path in PathTable:
                if agent.ID == agent_id:
                    GbestP = np.array([path[0].x,path[0].y]) if path[0] is not None else Gbest
                    distP = -1 if path is not None else -3 #-3 means no path, so tell Agent randomly walk
                    RoundDownLinkSingle(agent, GbestP, distP)
                    PathCount += 1
        

        # print('PSOCount: ',PSODownCount)
        # print('PathCount: ',PathCount)
        # DownLinkCount = PathCount + PSODownCount
        # print('DownLinkCount: ',DownLinkCount)
        
        # if len(AchievedAgents) > AchievedNumber:
        #     AchievedNumber = len(AchievedAgents) #for counting UpLinkNumber

        LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]



        # #draw
        # continue_loop = draw_agents_and_goal(display, agents, Goal)

        # # 在调用函数后检查是否需要继续循环
        # if not continue_loop:
        #     break  # 退出循环

        # print('AchievedAgents number: ',len(AchievedAgents))
        RoundMoves = len(agents)-len(AchievedAgents)
        CovMoves += RoundMoves
        # print('RoundMoves',RoundMoves)
        # print('Loop number',i+1)
        # print('CovMoves',CovMoves)

        if len(AchievedAgents)>= 0.9 * len(agents) or i >=50000:
        # if len(AchievedAgents)>= 1 or i >=30000:
            break
    return UpLinkNumber,DownLinkNumber, i+1, CovMoves

#main Code
gridMap = create_grid_map('dilatedImg4orgMap.png')
display = create_grid_map('grid_map.png')
gridMapCenter = find_map_center(gridMap)
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

# AgentsNum = np.linspace(20,100,9).astype(int)
AgentsNum = range(60,110,10)

# aNum = 100
for aNum in AgentsNum:
    LoopUpLinkNumbers = []
    LoopDownLinkNumbers = []
    Loopi = []
    LoopCovMoves = []
    for j in range(300): #Here is the Monte Carlo Loop number
        print(f'Agents number {aNum}, Loop number {j}')
        UpLinkNumber,DownLinkNumber, i, CovMoves = Single(aNum,ValidIdx,Goal)
        print('UpLinkNumber',UpLinkNumber)
        print('DownLinkNumber',DownLinkNumber)
        LoopUpLinkNumbers.append(UpLinkNumber)
        LoopDownLinkNumbers.append(DownLinkNumber)
        Loopi.append(i)
        LoopCovMoves.append(CovMoves)
        # np.save(f'Output/Test.npy',LoopUpLinkNumbers)

    np.save(f'Output/01DT/DT-Agents{aNum}-LoopUpLinkNumber.npy',LoopUpLinkNumbers)
    np.save(f'Output/01DT/DT-Agents{aNum}-LoopDownLinkNumber.npy',LoopDownLinkNumbers)
    np.save(f'Output/01DT/DT-Agents{aNum}-LoopiNumber.npy',Loopi)
    np.save(f'Output/01DT/DT-Agents{aNum}-LoopCovMoves.npy',LoopCovMoves)







