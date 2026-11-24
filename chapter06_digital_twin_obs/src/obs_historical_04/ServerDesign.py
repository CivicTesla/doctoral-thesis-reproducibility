import Agent
import numpy as np
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning




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


def FindGbest(agents):
    # for agent in agents:
    #     current_dist = EuclideanDistance(agent.position, Goal)
    #     if current_dist < dist:
    #         dist = current_dist
    #         Gbest = agent.position
    return min(agents, key=lambda agent: agent.dist)
    

# def FindGbest(agents,Gbest,dist):
#     min_dist = float('inf')  # 初始化为正无穷大，确保第一个代理的距离小于正无穷大
#     for agent in agents:
#         current_dist = EuclideanDistance(agent.position, Goal)
#         if current_dist < min_dist:
#             min_dist = current_dist
#             min_dist_agent_position = agent.position

    return min_dist_agent_position, min_dist

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


def Classify(agents):
    # i = 0
    PSOagents = []
    PathAgents = []
    PathKeepAgents = []
    for agent in agents:
        Udist = agent.dist
        # print('Udist',Udist)
        # i += 1
        # print('the i the time ',i)
        # if Udist == -3: #here means agent achived first element of path
           
        if Udist >= 0:
            PSOagents.append(agent)
        if Udist ==-1:
            # path = SesamePathPlanning(gridMap,agent.position,LGbest)#here Gbest!!!TODO!!!!!
            # # path = SesamePathPlanning(gridMap,agent.position,GbestU)
            # if path == None:
            #     print('path == None!!!!!!!!')
            # HistoryPath.append([agent.ID,path])
            PathAgents.append(agent)
        if Udist == -2 and agent.ID not in AchievedAgents:
           AchievedAgents.append(agent.ID)
    return PSOagents,PathAgents,AchievedAgents

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
        if PathTable:
            if Udist ==-1 and any(item[0] == agent.ID for item in PathTable):
        	    PathKeepAgents.append(agent)
        elif Udist == -1:
            PathAgents.append(agent)
        if Udist == -2 and agent.ID not in AchievedAgents:
           AchievedAgents.append(agent.ID)
    return PSOagents,PathAgents,AchievedAgents,PathKeepAgents



def PathAssignments(agents,Gbest,dist):# for agent.dist = -1
	PathTable = []
	for agent in agents:
		# print('path check for Gbest',Gbest)
		# print('agent ID',agent.ID)
		# print('agent position',agent.position)
		path = SesamePathPlanning(gridMap,agent.position,Gbest)
		path = path[1:] # the first element is agent's position
		if len(path)>=3:
			path=path[:-2] # To let agent get so PSO experience
		if path == None:
			print('path == None !!!!!!!!!')
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
from Image2gridMap import create_grid_map,get_valid_from_whole_map
from InitFunction import create_agents

gridMap = create_grid_map('org_map.png')

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([300,400])

N = 20

agents = create_agents(N, ValidIdx,Goal)

Uagents = RoundUpLinkSummarization(agents) #firt uplink agent's position and dist

print('length of initial uplink agents',len(Uagents))

best_agent = FindGbest(Uagents) # initial FindGbest
Gbest,dist = best_agent.position,best_agent.dist
dist = -1

DownLinkCount = RoundDownLinkSummarization(Uagents,Gbest,dist)

print('Initial DownLinkCount',DownLinkCount)

print('initial Gbest and dist',best_agent.position,best_agent.dist)


AchievedAgents = []
Uagents = RoundUpLinkSummarization(agents)#uplink successfl
# best_agent = FindGbest(Uagents) #PSO Gbest and dist

print('length of first looop uplink agents',len(Uagents))



PathTable = None #here because for the beginning there's no PathTable
PSOagents,PathAgents,AchievedAgents,PathKeepAgents = ClassifyNew(Uagents)
print('PathAgents number',len(PathAgents))
print('PSOagents number',len(PSOagents))
print('AchievedAgents number',len(AchievedAgents))
# PSOagents1,PathAgents1,AchievedAgents1,PathKeepAgents1 = ClassifyNew(Uagents)
print('PathKeepAgents number',len(PathKeepAgents))
for agentID in AchievedAgents:
	print('AchievedAgent ID: ', agentID)

# best_agent = FindGbest(PSOagents)

# Gbest, dist = best_agent1.position, best_agent1.dist

PathTableOld = PathAssignments(PathKeepAgents,Gbest,dist)
# print('PathTable length',len(PathTable))


# for item in PathTable:
#     agent_id = item[0]
#     path_nodes = item[1]
#     path_coordinates = [(node.x, node.y) for node in path_nodes]
#     print(f"Agent ID: {agent_id}, Path: {path_coordinates}")



# # for item in PathTable:
#     agent_id = item[0]
#     second_element_of_path = item[1][1]
#     if second_element_of_path is not None:
#         x_coord = second_element_of_path.x
#         y_coord = second_element_of_path.y
#         print(f"Agent ID: {agent_id}, Second Element Coordinates: ({x_coord}, {y_coord})")
#     else:
#         print(f"Agent ID: {agent_id}, Second Element of Path is None")


# item in PathTable: item[1][1].x   item[1][1].y are the second element of path !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# for agent in Uagents:
# 	agent_exists = any(item[0] == agent.ID for item in PathTable)
# 	if agent_exists:


# DownCountPath = 0
# for agent in PathAgents:
#     DownCountPath += 1 #no matter successful or failedS
#     #find corresponding path
#     matching_path = next((item[1] for item in PathTable if item[0] == agent.ID), None)

#     if matching_path is not None:
#         print(f"Path for Agent ID {agent.ID}: {matching_path[1].x,matching_path[1].y}")
#         # print(f"Path for Agent ID {agent.ID}: {matching_path}")
#         RoundDownLink4PathPlanning(agent, np.array([matching_path[1].x,matching_path[1].y]))
#         print('path length',len(matching_path))
#     else:
#     	agnet.AConnection(None,-3)# there's no path, so tell agent randomly walk


print('PathTable original length',len(PathTableOld))
DownCountPath = 0
index_to_remove = []  # Store indices to remove from PathTable

for agent in PathKeepAgents:
    DownCountPath += 1  # no matter successful or failed

    # find corresponding path
    matching_path_index = next((index for index, item in enumerate(PathTableOld) if item[0] == agent.ID), None)

    if matching_path_index is not None:
        matching_path = PathTableOld[matching_path_index][1]
        if len(matching_path) < 3:
            RoundDownLink4PathPlanning(agent, np.array([matching_path[0].x, matching_path[0].y]))
            index_to_remove.append(matching_path_index)
#             print('path length', len(matching_path))
        RoundDownLink4PathPlanning(agent, np.array([matching_path[0].x, matching_path[0].y]))
    else:
    	agnet.AConnection(None,-3)# there's no path, so tell agent randomly walk
 
# Remove items from PathTable
for index in reversed(index_to_remove):
    del PathTableOld[index]

print('PathTable After removal length',len(PathTableOld))



# def DownLinkPathPlanningSmooth(PathAgents):
    
DownCountPSO = RoundDownLinkSummarization(PSOagents,Gbest,dist)
print('DownPSO count: ',DownCountPSO)
print('DownPath count: ',DownCountPath)




# Gbest, dist definition before RoundDownLinkSummarization
for agent in PathKeepAgents:
	

    matching_path_index = next((index for index, item in enumerate(PathTableOld) if item[0] == agent.ID), None)
    # if matching_path_index is not None and 
    # matching_path = PathTableOld[matching_path_index][1]
    if matching_path_index is not None and isinstance(matching_path_index, int) and 0 <= matching_path_index < len(PathTableOld):
        matching_path = PathTableOld[matching_path_index][1]
        # 继续你的代码
    else:
        print("Matching path index is not a valid integer or is out of range.")
    if agent.dist == -3:
        # find corresponding path
        Gbest = np.array([matching_path[0].x,matching_path[0].y])

    # if dist == -1 and any(item[0] == agent.ID for item in PathTableOld):
    if dist == -1:
        path = matching_path[1:] # -1 means achieved, so eliminate the first element
        Gbest = np.array([path[0].x,path[0].y])









