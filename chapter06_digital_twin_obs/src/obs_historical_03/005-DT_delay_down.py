import Agent
import numpy as np
import os
import csv
import json
from ComRate import ComPerMap,ComLogic
from SesamePathPlanning import SesamePathPlanning
import cv2
from InitFunction import create_agents
from Image2gridMap import create_grid_map,get_valid_from_whole_map
Epsilon = 5
DEBUG_INVALID_POSITION = os.environ.get("DEBUG_INVALID_POSITION") == "1"

def ValidateAgentPositions(stage, agents):
    if not DEBUG_INVALID_POSITION:
        return
    for agent in agents:
        cell = agent.position.astype(int)
        if (
            cell[0] < 0
            or cell[0] >= gridMap.shape[0]
            or cell[1] < 0
            or cell[1] >= gridMap.shape[1]
            or gridMap[cell[0], cell[1]] != 0
        ):
            raise RuntimeError(
                f"{stage}: invalid agent position, "
                f"agent ID {agent.ID}, position {agent.position}, "
                f"cell {cell}, grid value "
                f"{gridMap[cell[0], cell[1]] if 0 <= cell[0] < gridMap.shape[0] and 0 <= cell[1] < gridMap.shape[1] else 'out-of-bounds'}"
            )

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
    achieved_agents = [agent for agent in agents if agent.dist == -2]
    if achieved_agents:
        return achieved_agents[0].position,Epsilon #because now the dist is a flag

    valid_agents = [agent for agent in agents if agent.dist > 0]
    if valid_agents:
        bestAgent = min(valid_agents, key=lambda agent: agent.dist)
        if bestAgent.dist < dist:
            return bestAgent.position,bestAgent.dist
    return  Gbest,dist


def RecordAgentHistory(agents, position_history, dist_history):
    for agent in agents:
        position_history[agent.ID].append(agent.position.copy())
        dist_history[agent.ID].append(agent.dist)


def FindGbestDelayed(agents, Gbest, dist, position_history, dist_history, current_round, tau):
    if agents:
        delayed_index = max(0, current_round - tau)
        best_delayed_dist = None
        best_delayed_position = None

        for agent in agents:
            agent_dist_history = dist_history[agent.ID]
            agent_position_history = position_history[agent.ID]

            if not agent_dist_history:
                continue

            history_index = min(delayed_index, len(agent_dist_history) - 1)
            delayed_dist = agent_dist_history[history_index]
            delayed_position = agent_position_history[history_index]

            if delayed_dist != -2 and delayed_dist <= 0:
                continue

            if best_delayed_dist is None or delayed_dist < best_delayed_dist:
                best_delayed_dist = delayed_dist
                best_delayed_position = delayed_position

        if best_delayed_dist is None:
            return Gbest, dist
        if best_delayed_dist == -2:
            return best_delayed_position, Epsilon #because now the dist is a flag
        if 0 < best_delayed_dist < dist:
            return best_delayed_position, best_delayed_dist
    return Gbest, dist
    


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


def MergeAchievedAgents(achieved_agents, new_achieved_agents):
    achieved_by_id = {agent.ID: agent for agent in achieved_agents}
    for agent in new_achieved_agents:
        achieved_by_id[agent.ID] = agent
    return list(achieved_by_id.values())


def PrintFailureDiagnostics(agents, achieved_agents, Gbest, HistoryGbest, Historydist, i, tau):
    achieved_ids = {agent.ID for agent in achieved_agents}
    left_agents = [agent for agent in agents if agent.ID not in achieved_ids]

    print(
        f"FAILED tau={tau}, round={i}: "
        f"achieved={len(achieved_ids)}/{len(agents)}, "
        f"left={len(left_agents)}, Gbest={Gbest}, "
        f"HistoryGbest={HistoryGbest}, Historydist={Historydist}",
        flush=True,
    )
    for agent in left_agents:
        cell = agent.position.astype(int)
        grid_value = (
            gridMap[cell[0], cell[1]]
            if 0 <= cell[0] < gridMap.shape[0] and 0 <= cell[1] < gridMap.shape[1]
            else "out-of-bounds"
        )
        print(
            f"  left ID={agent.ID}, dist={agent.dist}, "
            f"lastValidDist={getattr(agent, 'lastValidDist', None)}, "
            f"position={agent.position}, cell={cell}, grid={grid_value}, "
            f"pBestDis={agent.pBestDis}, gBestDis={agent.gBestDis}, "
            f"localGbest={agent.Gbest}",
            flush=True,
        )




def PathAssignments(agents,Gbest):# for agent.dist = -1
    PathTable = []
    for agent in agents:
        agent.RepairInvalidPosition()
        ValidateAgentPositions("before PathAssignments", [agent])
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


def BuildSummaryRow(aNum, tau, LoopUpLinkNumbers, LoopDownLinkNumbers, Loopi, LoopCovMoves, LoopSuccess):
    rounds = np.array(Loopi, dtype=float)
    covmoves = np.array(LoopCovMoves, dtype=float)
    uplinks = np.array(LoopUpLinkNumbers, dtype=float)
    downlinks = np.array(LoopDownLinkNumbers, dtype=float)
    successes = np.array(LoopSuccess, dtype=float)

    return {
        'agents': int(aNum),
        'tau': int(tau),
        'runs': int(len(Loopi)),
        'success_rate': float(np.mean(successes)),
        'rounds_mean': float(np.mean(rounds)),
        'rounds_median': float(np.median(rounds)),
        'rounds_std': float(np.std(rounds)),
        'rounds_q25': float(np.percentile(rounds, 25)),
        'rounds_q75': float(np.percentile(rounds, 75)),
        'covmoves_mean': float(np.mean(covmoves)),
        'covmoves_median': float(np.median(covmoves)),
        'covmoves_std': float(np.std(covmoves)),
        'covmoves_q25': float(np.percentile(covmoves, 25)),
        'covmoves_q75': float(np.percentile(covmoves, 75)),
        'uplink_mean': float(np.mean(uplinks)),
        'uplink_median': float(np.median(uplinks)),
        'downlink_mean': float(np.mean(downlinks)),
        'downlink_median': float(np.median(downlinks)),
    }


#Main Code ###########################################################





def Single(N,ValidIdx,Goal,tau=0):
    agents = create_agents(N, ValidIdx,Goal)
    ValidateAgentPositions("after create_agents", agents)
    position_history = {agent.ID: [] for agent in agents}
    dist_history = {agent.ID: [] for agent in agents}
    current_round = 0

    UpLinkNumber = len(agents)#for the first up link summary

    Uagents = RoundUpLinkSummarization(agents) #first uplink agent's position and dist
    ValidateAgentPositions("after initial RoundUpLinkSummarization", agents)
    RecordAgentHistory(agents, position_history, dist_history)

    DownLinkNumber = len(Uagents) #first downlink number

    # print('number of initial uplink agents',len(Uagents))

    Gbest,dist = FindGbestDelayed(Uagents,None,np.inf,position_history,dist_history,current_round,tau) # initial FindGbest
    # dist = -1
    CovMoves = 0 #here to set a initial sum of RoundMoves

    HistoryGbest,Historydist = Gbest,dist

    DownLinkCount = RoundDownLinkSummarization(Uagents,Gbest,dist)
    ValidateAgentPositions("after initial RoundDownLinkSummarization", agents)

    # print('Initial DownLinkCount',DownLinkCount)

    # print('initial Gbest and dist',HistoryGbest,Historydist)

    #First Loop Uplink#######################################################
    current_round += 1
    Uagents = RoundUpLinkSummarization(agents)#uplink successfl
    ValidateAgentPositions("after first loop RoundUpLinkSummarization", agents)
    RecordAgentHistory(agents, position_history, dist_history)
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
        Gbest,dist = FindGbestDelayed(AchievedAgents,HistoryGbest,Historydist,position_history,dist_history,current_round,tau)
    else:
        Gbest,dist = FindGbestDelayed(PSOagents,HistoryGbest,Historydist,position_history,dist_history,current_round,tau)
        

    #first Downlink############################################################
    #for PSO downlink
    PSODownCount = RoundDownLinkSummarization(PSOagents,Gbest,dist)
    ValidateAgentPositions("after first PSO RoundDownLinkSummarization", agents)
    PathCount = 0

    PathTable = PathAssignments(PathAgents,Gbest)


    for agent in PathAgents:
        for agent_id, path in PathTable:
            if agent.ID == agent_id:
                GbestP = np.array([path[0].x,path[0].y]) if path[0] is not None else Gbest
                distP = -1 if path is not None else -3 #-3 means no path, so tell Agent randomly walk
                RoundDownLinkSingle(agent, GbestP, distP)
                ValidateAgentPositions("after first path RoundDownLinkSingle", agents)
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
        current_round += 1
        Uagents = RoundUpLinkSummarization(LeftAgents) 
        ValidateAgentPositions("after loop RoundUpLinkSummarization", agents)
        RecordAgentHistory(agents, position_history, dist_history)
        # print('number of Big loop uplink agents',len(Uagents))
        # if not PathTableOld :
        #     PathTableOld = []#here because for the beginning there's no PathTable
            # print('PathOld is None~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        AchievedAgentsU = CheckAchieved(Uagents)
        UagentsA = [agent for agent in Uagents if agent not in AchievedAgentsU]#uplink agents wating for assignments
        
        # 将 AchievedAgentsU 中的代理合并到 AchievedAgents 中
        AchievedAgents = MergeAchievedAgents(AchievedAgents, AchievedAgentsU)

        PSOagents,PathAgents = ClassifyPathPSO(UagentsA)
        # print('PathAgents number',len(PathAgents))
        # print('PSOagents number',len(PSOagents))
        # print('AchievedAgents number',len(AchievedAgents))

        #keep Gbest and dist are the best
        if AchievedAgents:
            Gbest,dist = FindGbestDelayed(AchievedAgents,HistoryGbest,Historydist,position_history,dist_history,current_round,tau)
        else:
            Gbest,dist = FindGbestDelayed(PSOagents,HistoryGbest,Historydist,position_history,dist_history,current_round,tau)

        if dist < Historydist:
            HistoryGbest,Historydist = Gbest,dist
        Gbest, dist = HistoryGbest, Historydist 

        
        UpLinkNumber += len(LeftAgents)

        DownLinkNumber += len(UagentsA)
        # print(f'UpLinkNumber: {UpLinkNumber}, DownLinkNumber: {DownLinkNumber}~~~~~~~~~~~~~~~~~')

        #Downlink ####################################################################
        #for PSO downlink##############
        PSODownCount = RoundDownLinkSummarization(PSOagents,HistoryGbest,Historydist)
        ValidateAgentPositions("after loop PSO RoundDownLinkSummarization", agents)
        PathCount = 0

        # PathAgents downlink##########
        PathTable = PathAssignments(PathAgents,Gbest)

        for agent in PathAgents:
            for agent_id, path in PathTable:
                if agent.ID == agent_id:
                    GbestP = np.array([path[0].x,path[0].y]) if path[0] is not None else Gbest
                    distP = -1 if path is not None else -3 #-3 means no path, so tell Agent randomly walk
                    RoundDownLinkSingle(agent, GbestP, distP)
                    ValidateAgentPositions("after loop path RoundDownLinkSingle", agents)
                    PathCount += 1
        

        # print('PSOCount: ',PSODownCount)
        # print('PathCount: ',PathCount)
        # DownLinkCount = PathCount + PSODownCount
        # print('DownLinkCount: ',DownLinkCount)
        
        # if len(AchievedAgents) > AchievedNumber:
        #     AchievedNumber = len(AchievedAgents) #for counting UpLinkNumber

        achieved_ids = {agent.ID for agent in AchievedAgents}
        LeftAgents = [agent for agent in agents if agent.ID not in achieved_ids]



        # #draw
        # continue_loop = draw_agents_and_goal(display, agents, Goal)

        # # 在调用函数后检查是否需要继续循环
        # if not continue_loop:
        #     break  # 退出循环
        # print('AchievedAgents number: ',len(AchievedAgents))
        RoundMoves = len(agents)-len(achieved_ids)
        CovMoves += RoundMoves
        # print('RoundMoves',RoundMoves)
        # print('Loop number',i+1)
        # print('CovMoves',CovMoves)

        if len(achieved_ids)>= 0.9 * len(agents) or i >=2000:
            break
    achieved_ids = {agent.ID for agent in AchievedAgents}
    success = 1 if len(achieved_ids) >= 0.9 * len(agents) else 0
    if success == 0:
        PrintFailureDiagnostics(agents, AchievedAgents, Gbest, HistoryGbest, Historydist, i, tau)
    return UpLinkNumber,DownLinkNumber, i+1, CovMoves, success

#main Code
gridMap = create_grid_map('dilatedImg4orgMap.png')
display = create_grid_map('grid_map.png')
gridMapCenter = find_map_center(gridMap)
# print('grid_map center',gridMapCenter)

ValidIdx = get_valid_from_whole_map(gridMap)

Goal = np.array([400,300])

# AgentsNum = [20, 40, 60, 80, 100]
AgentsNum = [60,80,100]
# test AgentsNum
# AgentsNum = [40]
tau_list = [0, 1, 2, 3, 4, 5]
# AgentsNum = range(100,110,10)

# aNum = 100
os.makedirs('Output/DTDelay', exist_ok=True)

summary_columns = [
    'agents',
    'tau',
    'runs',
    'success_rate',
    'rounds_mean',
    'rounds_median',
    'rounds_std',
    'rounds_q25',
    'rounds_q75',
    'covmoves_mean',
    'covmoves_median',
    'covmoves_std',
    'covmoves_q25',
    'covmoves_q75',
    'uplink_mean',
    'uplink_median',
    'downlink_mean',
    'downlink_median',
]
summary_rows = []

for aNum in AgentsNum:
    for tau in tau_list:
        LoopUpLinkNumbers = []
        LoopDownLinkNumbers = []
        Loopi = []
        LoopCovMoves = []
        LoopSuccess = []
        for j in range(100): #Here is the Monte Carlo Loop number
            print(f'Agents number {aNum}, tau {tau}, Loop number {j}', flush=True)
            UpLinkNumber,DownLinkNumber, i, CovMoves, success = Single(aNum,ValidIdx,Goal,tau=tau)
            print(
                f'Finished agents {aNum}, tau {tau}, Loop number {j}: '
                f'rounds={i}, covmoves={CovMoves}, success={success}',
                flush=True
            )
            # print('UpLinkNumber',UpLinkNumber)
            # print('DownLinkNumber',DownLinkNumber)
            LoopUpLinkNumbers.append(UpLinkNumber)
            LoopDownLinkNumbers.append(DownLinkNumber)
            Loopi.append(i)
            LoopCovMoves.append(CovMoves)
            LoopSuccess.append(success)
            # np.save(f'Output/Test.npy',LoopUpLinkNumbers)

        np.save(f'Output/DTDelay/DelayTau{tau}-Agents{aNum}-LoopUpLinkNumber.npy',LoopUpLinkNumbers)
        np.save(f'Output/DTDelay/DelayTau{tau}-Agents{aNum}-LoopDownLinkNumber.npy',LoopDownLinkNumbers)
        np.save(f'Output/DTDelay/DelayTau{tau}-Agents{aNum}-LoopiNumber.npy',Loopi)
        np.save(f'Output/DTDelay/DelayTau{tau}-Agents{aNum}-LoopCovMoves.npy',LoopCovMoves)
        np.save(f'Output/DTDelay/DelayTau{tau}-Agents{aNum}-Success.npy',LoopSuccess)

        summary_row = BuildSummaryRow(aNum, tau, LoopUpLinkNumbers, LoopDownLinkNumbers, Loopi, LoopCovMoves, LoopSuccess)
        summary_rows.append(summary_row)
        print(
            f"tau={tau}, success_rate={summary_row['success_rate']:.2f}, "
            f"median_rounds={summary_row['rounds_median']}, "
            f"median_covmoves={summary_row['covmoves_median']}",
            flush=True
        )

summary_csv_path = 'Output/DTDelay/DTDelay-summary.csv'
summary_json_path = 'Output/DTDelay/DTDelay-summary.json'

with open(summary_csv_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=summary_columns)
    writer.writeheader()
    writer.writerows(summary_rows)

with open(summary_json_path, 'w') as json_file:
    json.dump(summary_rows, json_file, indent=2)
