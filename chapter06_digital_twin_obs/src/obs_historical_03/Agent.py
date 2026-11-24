import numpy as np 
#import cv2
import random
from LocalSensor import find_coordinates_inside_circle
from CollisionCheck import no_obstacle_on_line
from MoveAlongAngle import move_along_angle

Goal = np.array([400,300])

from Image2gridMap import create_grid_map
gridMap = create_grid_map('dilatedImg4orgMap.png')
Epsilon = 5

class Agent():

    def __init__(self, agent_id, init_position, init_velocity,init_dist):
        self.position = init_position# Numpy array [x, y]
        self.velocity = init_velocity  # Numpy array [x, y]
        self.max_vel = 10
        self.pBestDis = np.Inf # personal best distance
        self.gBestDis = np.Inf
        self.pbestPos = init_position
        self.sensorRadius = 15
        self.CurrentDist = self.EuclideanDistance(self.position,Goal)
        self.ID = agent_id
        self.dist = init_dist
        self.Gbest = self.position
        self.lastValidDist = init_dist
        self.lastValidPosition = init_position

    def _record_valid_state(self):
        if self.dist > 0:
            self.lastValidDist = self.dist
            self.lastValidPosition = self.position.copy()
        elif self.dist == -2:
            self.lastValidDist = 0
            self.lastValidPosition = self.position.copy()

    @staticmethod
    def _is_valid_position(position):
        position_cell = position.astype(int)
        return (
            0 <= position_cell[0] < gridMap.shape[0]
            and 0 <= position_cell[1] < gridMap.shape[1]
            and gridMap[position_cell[0], position_cell[1]] == 0
        )

    def RepairInvalidPosition(self):
        if self._is_valid_position(self.position):
            return False

        current_cell = self.position.astype(int)
        for radius in range(1, self.sensorRadius * 4 + 1):
            row_min = max(0, current_cell[0] - radius)
            row_max = min(gridMap.shape[0], current_cell[0] + radius + 1)
            col_min = max(0, current_cell[1] - radius)
            col_max = min(gridMap.shape[1], current_cell[1] + radius + 1)

            free_cells = np.argwhere(gridMap[row_min:row_max, col_min:col_max] == 0)
            if len(free_cells) == 0:
                continue

            free_cells = free_cells + np.array([row_min, col_min])
            nearest = free_cells[
                np.argmin(np.linalg.norm(free_cells - self.position, axis=1))
            ]
            print(
                f"Repair invalid position for ID {self.ID}: "
                f"{self.position} -> {nearest}"
            )
            self.position = nearest.astype(float)
            self.dist = self.EuclideanDistance(self.position, Goal)
            self._record_valid_state()
            return True

        self.dist = -1
        return False

    def PSOvelocity(self,Gbest,dist): # using PSO to return position and distance
        self.RepairInvalidPosition()
        # CurrentDist = self.EuclideanDistance(self.position, Goal)
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return

        if np.allclose(Gbest, self.position) and self.dist > Epsilon:
            return self.RandomlyWalk()
        #PSO parameter
        c1=c2=2
        # print('befor PSO agent position is ',self.position)
        w = 0.9 #w represents the global search capability, with a larger value indicating a stronger search ability."
        # if self.dist < self.pBestDis:
        #     self.Gbest = self.position

        if self.dist < dist:
            self.gBestDis = self.dist
            self.Gbest = self.position
        else:
            self.Gbest = Gbest
        if self.dist < self.pBestDis:
            self.pBestDis = self.dist
            self.pbestPos = self.position
        if dist < self.gBestDis:
            self.gBestDis = dist
            # print('before position',self.position)
        # velocity = w * self.velocity + c1 * np.random.rand()*(self.pbestPos-self.position) + \
        #         c2 * np.random.rand() * (self.Gbest - self.position)+ np.random.normal(0,1)*self.max_vel
        

        velocity = w * self.velocity + c1 * np.random.rand()*(self.pbestPos-self.position) + \
                c2 * np.random.rand() * (self.Gbest - self.position)+ np.random.uniform(-1,1)*2


        velocity = self.VelocityLimit(velocity,self.max_vel)
        # print('velocity',velocity)
        position = self.position + velocity
        # print('before pso position: ',self.position)
    
        if (
            self._is_valid_position(position)
            and no_obstacle_on_line(gridMap, self.position,position)
        ):
            # print('no obstacle on line: ')
            # print('no no_obstacle_on_line',position)
            self.position = position
            self.velocity = velocity
            self.dist = self.EuclideanDistance(self.position,Goal)
            self._record_valid_state()
            if self.dist < self.pBestDis:
                self.pBestDis = self.dist
                self.pbestPos = self.position
            # print('successful PSO agent positionis ',self.position)
            # print('successful pso new position',self.position)
            if 0<= self.dist <= Epsilon: #means achieved Goal:
                self.dist = -2 # means achieved Goal
                self._record_valid_state()
                return self.position, self.dist
            return self.position,self.dist
        # print('failed PSO agent position is ',self.position)
        self.dist = -1
        # print('self.position after dist = -1',self.position)
        # print('pso self.position',self.position)
        return self.position, self.dist #means PSO result has collision,so ask for PathPlanning



             

    # def SesameVelocity(self, Step):
    #     velocity = Step-self.position
    #     # print('velocity in SesameVelocity',velocity)
    #     # print('type of velocity in SesameVelocity',type(velocity))
    #     if np.linalg.norm(velocity) > self.max_vel:
    #         velocity = self.VelocityLimit(velocity,self.max_vel).astype(int)
    #         new_position = (self.position + velocity).astype(int)

    #         # 检查新位置是否合法
    #         if gridMap[new_position[0], new_position[1]] != 1:
    #             self.position = new_position
    #             self.dist = self.EuclideanDistance(self.position,Goal)
    #             if self.dist < self.pBestDis:
    #                 self.pBestDis = self.dist
    #                 self.pbestPos = self.position
    #             if self.dist <= Epsilon:
    #                     self.dist = -2
    #         else:
    #             # 处理碰撞，这里可以根据需要添加适当的逻辑
    #             print("Collision detected at new position. Handling collision...")
    #             # 可以选择保持当前位置，或者采取其他处理方式
    #             self.dist = -1
    #         return self.position,self.dist

    #     else:
    #         self.position = Step
    #     # if np.array_equal(self.position.astype(int), Step): #check if get the Step position
    #     #     self.dist = -1 # means it need a new path planning point
    #         self.dist = self.EuclideanDistance(self.position,Goal)
    #         if self.dist < self.pBestDis:
    #             self.pBestDis = self.dist
    #             self.pbestPos = self.position
    #     # print('sesame position~~~~~~~~~~',self.position)
    #     # print('sesame dist~~~~~~~~~~~~~~',self.dist)
    #     if self.dist <= Epsilon:#Check if positions get better than Gbest
    #         self.Gbest = self.position
    #         self.dist = -2 # means achieved Goal
    #     # if self.dist <= 2: #means achieved Goal:
    #     #     self.dist = -2
    #     return self.position, self.dist 

    def SesameVelocity(self, Step):
        self.RepairInvalidPosition()
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return
        if not no_obstacle_on_line(gridMap,self.position,Step):#means Step is an obstacle
            print(f'Sesame has error!!!,Step is {Step}')
            print(f'corresponding agent position is {self.position}')
            # self.dist = -1 #ask for a new path
            self.RandomlyWalk()#just walk away to find new land
            return

        velocity = Step - self.position

        # 检查新位置是否合法
        new_position = self.position + velocity
        if self._is_valid_position(new_position):
            # 通过速度限制
            if np.linalg.norm(velocity) > self.max_vel:
                velocity = self.VelocityLimit(velocity, self.max_vel).astype(int)
                newPosition = self.position + velocity
                if self._is_valid_position(newPosition) and no_obstacle_on_line(gridMap,self.position,newPosition):
                    self.position = (self.position + velocity).astype(int)
                else:
                    self.dist = -1
                    return self.position, self.dist
            else:
                self.position = Step

            # 更新距离等信息
            self.dist = self.EuclideanDistance(self.position, Goal)
            self._record_valid_state()
            if self.dist < self.pBestDis:
                self.pBestDis = self.dist
                self.pbestPos = self.position

            if self.dist <= Epsilon:
                self.dist = -2  # 达到目标
                self._record_valid_state()
                self.Gbest = self.position  # 更新全局最优位置
        else:
            print("Collision detected at new position. Handling collision...")
            self.dist = -1  # 处理碰撞

        return self.position, self.dist

    def RandomlyWalk(self):
        self.RepairInvalidPosition()
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return
        if np.random.rand() < 0.5:
            AvailableCords = find_coordinates_inside_circle(self.position, self.sensorRadius, gridMap)
            if AvailableCords:
                newPosition = np.array(random.choice(AvailableCords))
                if no_obstacle_on_line(gridMap, self.position, newPosition):
                    velocity = self.VelocityLimit(newPosition - self.position, self.max_vel).astype(int)
                    nextPosition = self.position + velocity
                    if not self._is_valid_position(nextPosition):
                        self.dist = -1
                        return self.position, self.dist
                    self.position = nextPosition
                    self.dist = self.EuclideanDistance(self.position, Goal)
                    self._record_valid_state()
                    if self.dist < self.pBestDis:
                        self.pBestDis = self.dist
                        self.pbestPos = self.position
                    if self.dist < self.EuclideanDistance(self.Gbest, Goal):
                        self.Gbest = self.position
                    if self.dist <= Epsilon:
                        self.dist = -2
                        self._record_valid_state()
                else:
                    self.dist = -1

            else:
                print(f'ID {self.ID} has no AvailableCords!')
                print('agent position is: ', self.position)
                self.dist = -1  # ask for help, stay there

        else:
            velocity = self.VelocityLimit(self.Gbest - self.position,self.max_vel).astype(int)
            newPosition = self.position + velocity
            if self._is_valid_position(newPosition) and no_obstacle_on_line(gridMap, self.position, newPosition):
                self.position = newPosition
                self.velocity = velocity
                self.dist = self.EuclideanDistance(self.position, Goal)
                self._record_valid_state()
                if self.dist < self.pBestDis:
                    self.pBestDis = self.dist
                    self.pbestPos = self.position
                if self.dist < self.EuclideanDistance(self.Gbest, Goal):
                    self.Gbest = self.position
                if self.dist <= Epsilon:
                        self.dist = -2
                        self._record_valid_state()
            else:
                self.dist = -1
 
        return self.position, self.dist



            

    def AConnection(self,Gbest,dist):
        self.RepairInvalidPosition()
        # if self.dist > Epsilon and dist == -2:
        #     print(f'agent ID {agent.id}, position: {agent.position}')
        if dist== -3 : # downlink/uplink failed,randomly walk mode
            # Communication failed: first try PSO with the agent's cached Gbest.
            # Fall back to random walk only when PSO cannot make a valid move.
            best_dist = self.gBestDis if self.gBestDis > 0 else self.lastValidDist
            self.PSOvelocity(self.Gbest, best_dist)
            if self.dist == -1:
                self.RandomlyWalk()
            return self.position, self.dist
                

            # self.dist = -1 #ask for help, stay there
            # return self.position,self.dist
            
        if dist >= 0:
            # self.position, self.dist = self.PSOvelocity(Gbest,dist)
            self.PSOvelocity(Gbest,dist)
        
        elif dist == -1: #-1 means use PathPlanning Mode
            # self.position, self.dist = self.SesameVelocity(Gbest)
            self.SesameVelocity(Gbest)
            
        # elif 0<= self.dist< Epsilon:#epsilon for Goal
        #     self.dist = -2 #-2 means received downlink message and keep stay
            
        return self.position, self.dist #upload position and distance
        
        



    @staticmethod
    def EuclideanDistance(a,b): #point a and point b
        return np.linalg.norm(a-b)

    
    @staticmethod
    def VelocityLimit(a,max_vel):
        if np.linalg.norm(a) <= max_vel:
            return a
        elif np.linalg.norm(a)> max_vel:
            e = np.linalg.norm(a)
            a = a.astype(float)/e
            a *= max_vel 
        return a

    # #TODO:
    # def Interface(self,Gbest,dist):


    

# # test Code##################################################################
# Goal = np.array([100, 100])

# from Image2gridMap import create_grid_map
# gridMap = create_grid_map('org_map.png')
# # gridMap = np.zeros((200, 200))  # 根据实际情况定义 gridMap

# # 创建 Agent 实例
# init_position = np.array([50, 50])
# init_velocity = np.array([1, 1])
# agent = Agent(init_position, init_velocity)

# # 运行测试
# CommFlag = 1  # 设置 CommFlag 的值，根据实际需要修改
# Gbest = np.array([80, 80])
# dist = -2#here can check 3 mode: dist >0, dist =-1, dist = -2

# # 测试 AConnection 方法
# new_position, result_flag = agent.AConnection(CommFlag, Gbest, dist)
# print("New Position:", new_position)
# print("Result Flag:", result_flag)
