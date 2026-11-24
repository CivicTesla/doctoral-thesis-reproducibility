import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning
import cv2
from Image2gridMap import create_grid_map,get_valid_from_whole_map
from InitFunction import create_agents

def find_map_center(grid_map):
    num_rows = len(grid_map)
    num_cols = len(grid_map[0])

    center_row = num_rows // 2
    center_col = num_cols // 2

    return center_row, center_col

def EuclideanDistance(a,b): #point a and point b
    return np.linalg.norm(a-b)

def RoundUpLinkSummarization(agents):
	UpLinkCount = 0
	UpLinkagents = []
	for agent in agents:
		UpLinkCount += 1 # no matter successful or fail, uplink tried once
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
        if bestAgent.dist < dist:
            return bestAgent.position,bestAgent.dist
    return  Gbest,dist
    

# def FindGbest(agents,Gbest,dist):
#     min_dist = float('inf')  # 初始化为正无穷大，确保第一个代理的距离小于正无穷大
#     for agent in agents:
#         current_dist = EuclideanDistance(agent.position, Goal)
#         if current_dist < min_dist:
#             min_dist = current_dist
#             min_dist_agent_position = agent.position

    # return min_dist_agent_position, min_dist

# def RoundDownLinkSummarizationInitial(agents,Gbest,dist):
# 	DownLinkCount = 0
# 	DownLinkagents = []
# 	for agent in agents:
# 		DownLinkCount += 1 # no matter successful or fail, uplink tried once
# 		if ComPerMap(agent.position):
# 			agent.AConnection(Gbest,dist)###########结果保留下来
# 			# print('real one downlink')
# 		else: # tell agent randomly walk
# 		    agent.AConnection(None,-3)
# 	return DownLinkCount


# def Classify(agents):
#     # i = 0
#     PSOagents = []
#     PathAgents = []
#     PathKeepAgents = []
#     for agent in agents:
#         Udist = agent.dist
#         # print('Udist',Udist)
#         # i += 1
#         # print('the i the time ',i)
#         # if Udist == -3: #here means agent achived first element of path
           
#         if Udist >= 0:
#             PSOagents.append(agent)
#         if Udist ==-1:
#             # path = SesamePathPlanning(gridMap,agent.position,LGbest)#here Gbest!!!TODO!!!!!
#             # # path = SesamePathPlanning(gridMap,agent.position,GbestU)
#             # if path == None:
#             #     print('path == None!!!!!!!!')
#             # HistoryPath.append([agent.ID,path])
#             PathAgents.append(agent)
#         if Udist == -2 and agent.ID not in AchievedAgents:
#            AchievedAgents.append(agent.ID)
#     return PSOagents,PathAgents,AchievedAgents

def ClassifyNew(agents):
    # i = 0
    PSOagents = []
    PathAgents = []
    PathKeepAgents = []
    for agent in agents:
        # UGbest,Udist = agent.AConnection(Gbest,dist)
        Udist = agent.dist
        # print('Udist',Udist)
        # i += 1
        # print('the i the time ',i)
        if Udist == -3: #here means agent achived first element of path
            PathKeepAgents.append(agent)
        if Udist >= 0:
            PSOagents.append(agent)
        if PathTableOld:
            if Udist ==-1 and any(item[0] == agent.ID for item in PathTableOld):
        	    PathKeepAgents.append(agent)
        elif Udist == -1:
            PathAgents.append(agent)
        # if Udist == -2 and agent.ID not in AchievedAgents:
        #    AchievedAgents.append(agent.ID)
        # 检查 AchievedAgents 是否已定义
        if 'AchievedAgents' not in locals():
            AchievedAgents = []

        # 检查 Udist 是否等于 -2
        if Udist == -2:
            # 检查 agent.ID 是否在 AchievedAgents 中
            if agent not in AchievedAgents:
                AchievedAgents.append(agent)
    return PSOagents,PathAgents,AchievedAgents,PathKeepAgents



def PathAssignments(agents,Gbest):# for agent.dist = -1
    PathTable = []
    for agent in agents:
        # print('path check for Gbest',Gbest)
        # print('agent ID',agent.ID)
        # print('agent position',agent.position)
        path = SesamePathPlanning(gridMap,agent.position,Gbest)
        if path is not None:
            if len(path)>=3:
                path=path[:-2] # To let agent get so PSO experience
            path = path[1:] # the first element is agent's position
        
        if path == None:
            print('path == None !!!!!!!!!')
            print(f'ID {agent.ID} position is {agent.position}!')

            continue    
        PathTable.append((agent.ID,path))

    return PathTable

def RoundDownLink4PathPlanning(agent,Gbest):
	if ComPerMap(agent.position):
			agent.AConnection(Gbest,-1)
			# print('real one downlink')
	else: # tell agent randomly walk,because downlink failed
		    agent.AConnection(None,-3)


def RoundDownLinkSummarization(agents,Gbest,dist):
	DownLinkCount = 0
	DownLinkagents = []
	for agent in agents:
		DownLinkCount += 1 # no matter successful or fail, uplink tried once
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


#Test Code ###########################################################


gridMap = create_grid_map('dilatedImg4orgMap.png')
# gridMapCenter = find_map_center(gridMap)
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

N = 20

agents = create_agents(N, ValidIdx,Goal)

#check agents positions
for agent in agents:
    print(f'ID {agent.ID} \'s position is {agent.position}')

Uagents = RoundUpLinkSummarization(agents) #firt uplink agent's position and dist

print('length of initial uplink agents',len(Uagents))

Gbest,dist = FindGbest(Uagents,None,np.Inf) # initial FindGbest
# dist = -1


#check agents positions
for agent in agents:
    print(f'ID {agent.ID} \'s position is {agent.position}')

HistoryGbest,Historydist = Gbest,dist

DownLinkCount = RoundDownLinkSummarization(Uagents,Gbest,dist)

print('Initial DownLinkCount',DownLinkCount)

print('initial Gbest and dist',HistoryGbest,Historydist)

#First Loop Uplink#######################################################
Uagents = RoundUpLinkSummarization(agents)#uplink successfl
# best_agent = FindGbest(Uagents) #PSO Gbest and dist

#check agents positions
for agent in agents:
    print(f'ID {agent.ID} \'s position is {agent.position}')

print('length of first loop uplink agents',len(Uagents))



PathTableOld = [] #here because for the beginning there's no PathTable
PSOagents,PathAgents,AchievedAgents,PathKeepAgents = ClassifyNew(Uagents)
print('PathAgents number',len(PathAgents))
print('PSOagents number',len(PSOagents))
print('AchievedAgents number',len(AchievedAgents))
# PSOagents1,PathAgents1,AchievedAgents1,PathKeepAgents1 = ClassifyNew(Uagents)
print('PathKeepAgents number',len(PathKeepAgents))
for agentID in AchievedAgents:
	print('AchievedAgent ID: ', agentID)

#keep Gbest and dist are the best
if AchievedAgents:
    Gbest,dist = AchievedAgents[0].position,AchievedAgents[0].dist
else:
    Gbest,dist = FindGbest(PSOagents,HistoryGbest,Historydist)
    

#generate first PathTable if have PathAgents
if PathAgents:
    PathTableOld = PathAssignments(PathAgents,Gbest)
    print('initial PathTable length',len(PathTableOld))

else:
    PathTableOld = []
    # print('No PathTable!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#first Downlink############################################################
#for PSO downlink
PSODownCount = RoundDownLinkSummarization(PSOagents,Gbest,dist)
PathCount = 0
# PathAgents downlink
for agent in PathAgents:
    matching_path_index = next((index for index, item in enumerate(PathTableOld) if item[0] == agent.ID), None)
    if matching_path_index:
        matching_path = PathTableOld[matching_path_index][1]
        # find corresponding path
        GbestP,distP = np.array([matching_path[0].x,matching_path[0].y]),-1
    else: # tell agent randomly walk,because path not found
        GbestP,distP = None,-3

    RoundDownLinkSingle(agent,GbestP, distP)
    PathCount += 1


print('PSOCount',PSODownCount)
print('PathCount',PathCount)
DownLinkCount = PathCount + PSODownCount





#Big Loop Start#######################################
for i in range(1000):
    print(f'the {i}th time for the loop!!!!!!!!!!!!!!')
    Uagents = RoundUpLinkSummarization(agents) 
    print('length of Big loop uplink agents',len(Uagents))
    
    print('PathTableOld: ',len(PathTableOld))
    if not PathTableOld :
        PathTableOld = []#here because for the beginning there's no PathTable
        # print('PathOld is None~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    PSOagents,PathAgents,AchievedAgents,PathKeepAgents = ClassifyNew(Uagents)
    print('PathAgents number',len(PathAgents))
    print('PSOagents number',len(PSOagents))
    print('AchievedAgents number',len(AchievedAgents))
    print('PathKeepAgents number',len(PathKeepAgents))
    for agentID in AchievedAgents:
        print('AchievedAgent ID: ', agentID)

    #keep Gbest and dist are the best
    if AchievedAgents:
        Gbest,dist = AchievedAgents[0].position,AchievedAgents[0].dist
    else:
        Gbest,dist = FindGbest(PSOagents,HistoryGbest,Historydist)

    if dist < Historydist:
        HistoryGbest,Historydist = Gbest,dist
    Gbest, dist = HistoryGbest, Historydist

    #Downlink ####################################################################
    #for PSO downlink##############
    PSODownCount = RoundDownLinkSummarization(PSOagents,HistoryGbest,Historydist)
    PathCount = 0

    # PathAgents downlink##########
    #generate first PathTable if have PathAgents
    PathTableNew = []
    if PathAgents:
        PathTableNew = PathAssignments(PathAgents,Gbest)
        if PathTableNew:
            for agent in PathAgents:
                matching_path_index = next((index for index, item in enumerate(PathTableNew) if item[0] == agent.ID), None)
                matching_path = PathTableNew[matching_path_index][1]
                # find corresponding path
                GbestP = np.array([matching_path[0].x,matching_path[0].y])
                RoundDownLinkSingle(agent,GbestP,-1)
                PathCount += 1
    # print('NewTable length~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',len(PathTableNew))

    print('PSO downlink Count',PSODownCount)
    print('PathCount',PathCount)
    

    if PathTableNew:
        PathTableOld += PathTableNew # update PathTableOld

    # PathKeep downlink##############
    PathKeepCount = 0
    if PathKeepAgents:
        print('length of PathTableOld',len(PathTableOld))
        for agent in PathKeepAgents:
            matching_path_index = next((index for index, item in enumerate(PathTableOld) if item[0] == agent.ID), None)
            matching_path = PathTableOld[matching_path_index][1]
            # print('agent.dist in PathKeepAgents',agent.dist)
            if agent.dist == -3:
                # find corresponding path
                GbestP,distP = np.array([matching_path[0].x,matching_path[0].y]),-1
                
            # if dist == -1 and any(item[0] == agent.ID for item in PathTableOld):
            if dist == -1 and len(matching_path) >=1:
                path = matching_path[1:] # -1 means achieved, so eliminate the first element
                if path == None:
                    GbestP,distP= Gbest,dist #means OldPath finishes,tell agent to get PSO mode
                else:
                    GbestP,distP = np.array([path[0].x,path[0].y]),-1
                PathTableOld[matching_path_index][1] = path #here update PathTable
            elif dist == -1:
                GbestP,distP= Gbest,dist

            RoundDownLinkSingle(agent,GbestP,distP)
            PathKeepCount += 1
        # # 如果 len(matching_path) 小于 3，删除对应的行
        # if len(matching_path) < 3:
        #     del PathTableOld[matching_path_index]
        #     continue
    
    DownLinkCount = PathCount + PSODownCount + PathKeepCount
    print('Path downlink Count',PathCount)
    print('DownLinkCount',DownLinkCount)

    #draw
    # cv2.drawMarker(display, Goal, (0,0,255), markerType=1,thickness = 6)
    # cv2.imshow('progress',display)
    # #dynamic show 27ms pro time
    # cv2.waitKey(27)
    # display[:,:,:] = cv2.imread('BackDis.png')
    display = cv2.imread('org_map.png')
    for agent in agents:
        cv2.circle(display, agent.position, 5, (90, 30, 60), -1, cv2.LINE_AA) 
        # cv2.circle(display, agent.position, 5, (255, 0, 0), -1) 
    cv2.drawMarker(display, Goal, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    # 显示图像
    cv2.imshow('Map with Agents and Goal', display)

    # 检测按键
    key = cv2.waitKey(100)  # 每100毫秒检查一次按键

    # 如果按下Esc键，退出循环
    if key == 27:
        break













