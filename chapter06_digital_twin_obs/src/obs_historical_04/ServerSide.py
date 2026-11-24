import numpy as np
import Agent
from ComRate import ComPerMap,ComLogic
from InitFunction import create_agents
from Image2gridMap import create_grid_map,get_valid_from_whole_map
from SesamePathPlanning import sample, find_path
from CollisionCheck import no_obstacle_on_line

def EuclideanDistance(a,b): #point a and point b
    return np.linalg.norm(a-b)



gridMap = create_grid_map('org_map.png')

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([300,400])

N = 50

agents = create_agents(N, ValidIdx)

dist = np.Inf

UpLinkCount = 0
DownLinkCount = 0

#get init positions and return Gbest and best dist


# print('final dist !!!!!!!: ',dist)
# def ConnectionCheck(agents):
# 	UplinkAgents = []
# 	for agent in agents:
# 		if ComLogic(agent,Gbest,dist): #if Uplink successful
# 			UplinkAgents.append(agent)
# 		else:
# 			agent.AConnection(np.array([np.Inf,np.Inf]),-3) #Uplink failed,tell agent get randomly walk
# 	return UplinkAgents



def ConnectionCheck(agents):
	UplinkAgents = []
	for agent in agents:
		if ComLogic(agent,Gbest,dist): #if Uplink successful
			UplinkAgents.append(agent)
		else:
			agent.AConnection(np.array([np.Inf,np.Inf]),-3) #Uplink failed,tell agent get randomly walk
	return UplinkAgents



# position,dist = agent.AConnection(Gbest,dist)

# def Interface(Gbest,dist):
def classify(agents,Gbest,dist): # here the agents are successfully uplink agents
	PSOdist = np.Inf
	PSOGbest = None
	for agent in agents:
	    if agent.CurrentDsit <=PSOdist:#find agents' best PSO Gbest and dist
	    	PSOdist = agent.CurrentDist
            PSOGbest = agent.position
    for agent in agents: # try to downlink
        if ComPerMap(agent.position): #If Downlink successful

HistoryPath = []
AchievedAgents = []

def SClassify(agent,GbestU,distU):
	global HistoryPath,AchievedAgents
	PSOagents = []
	# Gbest,dist = ComLogic(agent,GbestU,distU)#check connection for UpLink
	# UpLinkCount += 1
	if distU == -3:
		agent.AConnection(GbestU,distU)# means agents get no downlink, a kind of trigger
		#But no Downlink, because no clue of agent's position
	if distU >=0:
		PSOagents.append(agent) #agents who can participate PSO
	if distU == -1:
		path = SesamePathPlanning(gridMap,agent.position,GbestU)
		if path == None:
			print('path == None!!!!!!!!')
		HistoryPath.append([agent.ID,path])
	if dist == -2:
		AchievedAgents.append(agent.ID)
	return PSOagents




def findGbest4PSO(agents,dist): # here it is assumed that agents have successfully communicated
	# global dist
	for agent in agents: 
		# if no_obstacle_on_line(gridMap,agent.position)
		TempDist = EuclideanDistance(agent.position,Goal)
		if TempDist < dist:
			dist = TempDist
			Gbest = agent.position
			# print('dist',dist)
	return Gbest,dist


def RoundUpLinkSummarization(agents):
	UpLinkCount = 0
	UpLinkagents = []
	for agent in agents:
		UpLinkCount += 1 # no matter successful or fail, uplink tried once
		if ComPerMap(agent.position):
			Connectedagents.append(agent)
		else:
		    SClassify(agent,None,-3)
	return UpLinkagents

dist = np.Inf

UpLinkagents = RoundUpLinkSummarization(agents)#successful up linked Agents

PSOGbest,PSOdist = findGbest4PSO(UpLinkagents,dist)

if PSOdist < dist:
	dist = PSOdist
	Gbest = PSOGbest


for agent in UpLinkagents:
	PSOagents = []
	Gbest, dist = agent.AConnection(Gbest,dist)
	if dist >= 0:
		PSOagents.append(agent)
	




for agent in PSOagents:
	position,Udist = agent.AConnection(Gbest,dist)
	SClassify(agent,position,Udist)







		    









#Test Code##############################
# 获取最佳代理
hahaAgents = ConnectionCheck(agents)
Gbest,dist = findGbest(hahaAgents)


# 打印最佳代理的位置和距离
print('number of agents',len(agents))
print('number of hahaAgents',len(hahaAgents))
print(f"The best agent is at position: {Gbest}")
print(f"The distance to the goal is: {dist}")
