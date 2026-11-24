
# from Agent05 import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
import cv2
from InitFunction05 import create_agents
from Image2gridMap import create_grid_map,get_valid_from_whole_map
Epsilon = 5


def check_agent_positions(agents, previous_positions):
    not_moved_agents = []
    for agent, prev_pos in zip(agents, previous_positions):
        if np.array_equal(agent.position, prev_pos):
            not_moved_agents.append(agent.ID)
    return not_moved_agents

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
    valid_agents = [agent for agent in agents if agent.dist>=0 or agent.dist == -2]

    if valid_agents:
        bestAgent = min(valid_agents, key=lambda agent: agent.dist)
        if bestAgent.dist == -2:
            return bestAgent.position, Epsilon  # because now the dist is a flag
        if 0 <= bestAgent.dist < dist:
            return bestAgent.position, bestAgent.dist
    else:
        print('44 法外之地: ')
    return Gbest,dist
    


def ClassifyPathPSO(agents):
    i = 0
    PSOagents = []
    PathAgents = []
    for agent in agents:
        if agent.dist >= 2: #save some space for achieved agents
            PSOagents.append(agent) 
        if agent.dist == -1:
            PathAgents.append(agent)
            print(f'still get Sesame Help Request {i}')
            i += 1
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


def CheckSesame(agents):
    i = 0
    for agent in agents:
        if agent.dist == -1:
            print(f'here is a sesame help request!!!!!!!!!!!!!!!  {i}')
            i += 1

def RoundDownLink4PathPlanning(agent,Gbest):
	if ComPerMap(agent.position):
			agent.AConnection(Gbest,-1)
			# print('real one downlink')
	else: # tell agent randomly walk,because downlink failed
		    agent.AConnection(Gbest,-3)


def RoundDownLinkSummarization(agents,Gbest,dist):
    DownLinkCount = 0
    # print(f'134 down summarization dist is {dist}')
    DownLinkagents = []
    for agent in agents:
        # if agent.dist == -1:
        #     print(f'138 found!!!!!!!!')
        DownLinkCount += 1 # no matter successful or fail, uplink tried once
        if ComPerMap(agent.position):
            # print(f'141 Gbest is {Gbest}, dist is {dist}')
            # print(f'142 agent dist is {agent.dist}')
            agent.AConnection(Gbest,dist)
            # print(f'144 Gbest is {Gbest}, dist is {dist}')
            # print(f'145 agent dist is {agent.dist}')
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


#Main Code ############################################################################################################

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

    # print(f'205 dist is {dist}')
    # CheckSesame(Uagents)

    # print('Initial DownLinkCount',DownLinkCount)

    # print('initial Gbest and dist',HistoryGbest,Historydist)


    #First Loop Uplink#######################################################
    Uagents = RoundUpLinkSummarization(agents)#uplink successfl
    # best_agent = FindGbest(Uagents) #PSO Gbest and dist



    # print('number of first loop uplink agents',len(Uagents))
    AchievedAgents = CheckAchieved(Uagents)
    UagentsA = [agent for agent in Uagents if agent not in AchievedAgents]#uplink agents wating for assignments
    # DownLinkNumber += len(UagentsA)
    # UpLinkNumber += len(agents)
    # print(f'UpLinkNumber: {UpLinkNumber}, DownLinkNumber: {DownLinkNumber}~~~~~~~~~~~~~~~~~~~~~')

    # CheckSesame(UagentsA)

    # PSOagents,PathAgents = ClassifyPathPSO(UagentsA)
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
        AchievedNumber = len(AchievedAgents)
    else:
        Gbest,dist = FindGbest(UagentsA,HistoryGbest,Historydist)
        AchievedNumber = 0

    DownLinkNumber += RoundDownLinkSummarization(UagentsA,Gbest,dist)

    # previous_positions = [agent.position.copy() for agent in agents]
    #first Downlink############################################################

    LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]
    i = 0
    #Big Loop Start#######################################
    while(True):
        # Initialize previous_positions with the initial positions of the agents
        # Check for moved agents
        # not_moved_agents = check_agent_positions(agents, previous_positions)
        #
        # # Print the IDs of agents that have moved
        # if not_moved_agents:
        #     for aA in not_moved_agents:
        #         print(f'Agent {aA} has not moved')

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
            AchievedNumber = 0
        else:
            AchievedAgents.extend(AchievedAgentsU)
            AchievedNumber = len(AchievedAgents)

        # PSOagents,PathAgents = ClassifyPathPSO(UagentsA)
        # print('PathAgents number',len(PathAgents))
        # print('PSOagents number',len(PSOagents))
        # print('AchievedAgents number',len(AchievedAgents))

        #keep Gbest and dist are the best
        if AchievedAgents:
            Gbest,dist = FindGbest(AchievedAgents,HistoryGbest,Historydist)
            # Gbest, dist = AchievedAgents[0].position, AchievedAgents[0].dist
        else:
            Gbest,dist = FindGbest(UagentsA,HistoryGbest,Historydist)

        if dist < Historydist:
            HistoryGbest,Historydist = Gbest,dist
        Gbest, dist = HistoryGbest, Historydist 

        
        UpLinkNumber += len(LeftAgents)

        DownLinkNumber += RoundDownLinkSummarization(UagentsA,Gbest,dist)

        # print(f'UpLinkNumber: {UpLinkNumber}, DownLinkNumber: {DownLinkNumber}~~~~~~~~~~~~~~~~~')

        #Downlink ####################################################################
        #for PSO downlink##############
        # PSODownCount = RoundDownLinkSummarization(PSOagents,HistoryGbest,Historydist)
        # PathCount = 0

        # PathAgents downlink##########
        # PathTable = PathAssignments(PathAgents,Gbest)

        # for agent in PathAgents:
        #     for agent_id, path in PathTable:
        #         if agent.ID == agent_id:
        #             GbestP = np.array([path[0].x,path[0].y]) if path[0] is not None else Gbest
        #             distP = -1 if path is not None else -3 #-3 means no path, so tell Agent randomly walk
        #             RoundDownLinkSingle(agent, GbestP, distP)
        #             PathCount += 1
        

        # print('PSOCount: ',PSODownCount)
        # print('PathCount: ',PathCount)
        # DownLinkCount = PathCount + PSODownCount
        # print('DownLinkCount: ',DownLinkCount)
        
        # if len(AchievedAgents) > AchievedNumber:
        #     AchievedNumber = len(AchievedAgents) #for counting UpLinkNumber

        LeftAgents = [agent for agent in agents if AchievedAgents is None or agent not in AchievedAgents]



        # # #draw
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

        # if len(AchievedAgents)>= 0.9 * len(agents) or i >=30000:
        if UpLinkNumber>30000:
            CovRate = len(AchievedAgents)/len(agents)
        # if len(AchievedAgents)>= 1 or i >=30000:
            break

        # Update previous_positions for the next round
        # previous_positions = [agent.position.copy() for agent in agents]

    return UpLinkNumber,DownLinkNumber, i+1, CovMoves, CovRate

#main Code#######################################################################################################################
gridMap = create_grid_map('dilatedImg4orgMap.png')
display = create_grid_map('grid_map.png')
gridMapCenter = find_map_center(gridMap)
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

# AgentsNum = np.linspace(20,100,9).astype(int)
AgentsNum = range(10,110,10)

# aNum = 100
for aNum in AgentsNum:
    LoopUpLinkNumbers = []
    LoopDownLinkNumbers = []
    Loopi = []
    LoopCovMoves = []
    LoopCovRate = []
    for j in range(300): #Here is the Monte Carlo Loop number
        print(f'Agents number {aNum}, Loop number {j}')
        UpLinkNumber,DownLinkNumber, i, CovMoves, CovRate = Single(aNum,ValidIdx,Goal)
        print('UpLinkNumber',UpLinkNumber)
        print('DownLinkNumber',DownLinkNumber)
        LoopUpLinkNumbers.append(UpLinkNumber)
        LoopDownLinkNumbers.append(DownLinkNumber)
        Loopi.append(i)
        LoopCovMoves.append(CovMoves)
        LoopCovRate.append(CovRate)
        # np.save(f'Output/Test.npy',LoopUpLinkNumbers)

    np.save(f'Output/10CovRateDTnoObsvsP2P/DT-Agents{aNum}-LoopUpLinkNumber.npy',LoopUpLinkNumbers)
    np.save(f'Output/10CovRateDTnoObsvsP2P/DT-Agents{aNum}-LoopDownLinkNumber.npy',LoopDownLinkNumbers)
    np.save(f'Output/10CovRateDTnoObsvsP2P/DT-Agents{aNum}-LoopiNumber.npy',Loopi)
    np.save(f'Output/10CovRateDTnoObsvsP2P/DT-Agents{aNum}-LoopCovMoves.npy',LoopCovMoves)
    np.save(f'Output/10CovRateDTnoObsvsP2P/DT-Agents{aNum}-LoopCovRate.npy',LoopCovRate)










