#here is the "no obstacle mode", map has no dt, but agent has dt
import numpy as np 
#import cv2
import random
from LocalSensor import find_coordinates_inside_circle,find_coordinates_inside_ring
from CollisionCheck import no_obstacle_on_line
from MoveAlongAngle import move_along_angle


"""
        dist = -3 upload/download fails
        dist = -1 need PathPlanning Mode
        dist = -2 achieved
        dist >=0 calculated distance between agent and goal
        """


Goal = np.array([400,300])

from Image2gridMap import create_grid_map
gridMap = create_grid_map('dilatedImg4orgMap.png') #the gridMap is for local sensor
Epsilon = 5

class Agent():

    def __init__(self, agent_id, init_position, init_velocity,init_dist):
        self.position = init_position# Numpy array [x, y]
        self.velocity = init_velocity  # Numpy array [x, y]
        self.max_vel = 10
        self.pBestDis = np.Inf # personal best distance
        self.gBestDis = np.Inf
        self.pbestPos = init_position
        self.sensorRadius = 6
        self.CurrentDist = self.EuclideanDistance(self.position,Goal)
        self.ID = agent_id
        self.dist = init_dist
        self.Gbest = self.position


    # def PSOvelocity(self,Gbest,dist): # using PSO to return position and distance
    #     # CurrentDist = self.EuclideanDistance(self.position, Goal)
    #     # print(f'Agent{self.ID} 42 curent position is {self.position} PSO')
    #     self.dist = self.EuclideanDistance(self.position,Goal)
    #     if self.dist <= Epsilon:
    #         self.dist = -2
    #         return

    #     if np.any(Gbest == self.position) and self.dist > Epsilon:
    #         # print(f'49: Agent{self.ID}  curent position is {self.position} PSO to randomly walk')
    #         self.RandomlyWalk()
    #         # print(f'Agent{self.ID} 50 curent position is {self.position} PSO')
    #         return

    #     if self.dist <= dist:
    #         self.Gbest = self.pbestPos = self.position
    #         self.pBestDis = self.dist
    #         # print(f'Agent{self.ID} 58 curent position is {self.position} PSO to randomly walk')
    #         self.RandomlyWalk()

    #         return

    #     # if self.dist < self.pBestDis:
    #     #     self.Gbest = self.position

    #     # if self.dist > dist:
    #     #     self.Gbest = Gbest
    #     if self.dist < self.pBestDis:
    #         self.pBestDis = self.dist
    #         self.pbestPos = self.position
    #     if dist < self.gBestDis:
    #         self.gBestDis = dist
    #         self.Gbest = Gbest

    #         # print('before position',self.position)
    #     # velocity = w * self.velocity + c1 * np.random.rand()*(self.pbestPos-self.position) + \
    #     #         c2 * np.random.rand() * (self.Gbest - self.position)+ np.random.normal(0,1)*self.max_vel
    #     # PSO parameter
    #     c1 = c2 = 2
    #     w = 0.9  # w represents the global search capability, with a larger value indicating a stronger search ability."

    #     velocity = w * self.velocity + c1 * np.random.uniform(-1,1)*(self.pbestPos-self.position) + \
    #             c2 * np.random.uniform(-1,1) * (self.Gbest - self.position)+ np.random.uniform(-1,1)*2
    #     velocity = self.VelocityLimit(velocity,self.max_vel)
    #     # print('velocity',velocity)
    #     # position = self.position + velocity
    #     # # print(f'Agent{self.ID} 79 curent position is {self.position} PSO')
    #     # if no_obstacle_on_line(gridMap, self.position,position):
    #     #     # print('no obstacle on line: ')
    #     #     # print('no no_obstacle_on_line',position)
    #     #     self.position = position
    #     # 计算新位置（先用浮点数计算）
    #     new_position_float = self.position + velocity
    #     # 转换为整数位置
    #     new_position = np.round(new_position_float).astype(int)

    #     # 安全检查：验证新位置是否在地图边界内且不是障碍物
    #     is_valid_position = (
    #             0 <= new_position[0] < gridMap.shape[0] and
    #             0 <= new_position[1] < gridMap.shape[1] and
    #             gridMap[tuple(new_position)] == 0
    #     )

    #     if is_valid_position and no_obstacle_on_line(gridMap, self.position, new_position):
    #         self.position = new_position
    #         # print(f'Agent{self.ID} 89 curent position is {self.position} PSO')
    #         self.velocity = velocity
    #         self.dist = self.EuclideanDistance(self.position,Goal)
    #         if self.dist < self.pBestDis:
    #             self.pBestDis = self.dist
    #             self.pbestPos = self.position

    #         # print('successful PSO agent positionis ',self.position)
    #         # print('successful pso new position',self.position)

    #         if 0<= self.dist <= Epsilon: #means achieved Goal:
    #             self.dist = -2 # means achieved Goal
    #             # if self.dist == -1:
    #             #     print(f'Agent115 got -1 ')
    #             return self.position, self.dist
    #         return self.position,self.dist
    #     else:
    #         # print(f'108Agent {self.ID}, PSO not available, the corresponding position is {self.position}')
    #         self.RandomlyWalk()  # means if pso fails, agent needs to randomly walk
    #     # print('failed PSO agent position is ',self.position)
    #     # self.dist = -3 

    #     # if self.dist == -1:
    #     #     print(f'Agent119 got -1 ')
    #     # print('self.position after dist = -1',self.position)
    #     # print('pso self.position',self.position)
    #     # print(f'Agent{self.ID} 113 curent position is {self.position} PSO')
    #     return self.position, self.dist #means PSO result has collision,so ask for PathPlanning

    def PSOvelocity(self,Gbest,dist): # using PSO to return position and distance
        
        # CurrentDist = self.EuclideanDistance(self.position, Goal)
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return

        if np.any(Gbest == self.position) and self.dist > Epsilon:
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
        

        velocity = w * self.velocity + c1 * np.random.uniform(-1,1)*(self.pbestPos-self.position) + \
                c2 * np.random.uniform(-1,1) * (self.Gbest - self.position)+ np.random.uniform(-1,1)*2


        velocity = self.VelocityLimit(velocity,self.max_vel)
        # print('velocity',velocity)
        position = self.position + velocity
        # print('before pso position: ',self.position)
    
        if no_obstacle_on_line(gridMap, self.position,position):
            # print('no obstacle on line: ')
            # print('no no_obstacle_on_line',position)
            self.position = position
            self.velocity = velocity
            self.dist = self.EuclideanDistance(self.position,Goal)
            if self.dist < self.pBestDis:
                self.pBestDis = self.dist
                self.pbestPos = self.position
            # print('successful PSO agent positionis ',self.position)
            # print('successful pso new position',self.position)
            if 0<= self.dist <= Epsilon: #means achieved Goal:
                self.dist = -2 # means achieved Goal
                return self.position, self.dist
            return self.position,self.dist
        # print('failed PSO agent position is ',self.position)
        # self.dist = -1
        self.RandomlyWalk()
        # print('self.position after dist = -1',self.position)
        # print('pso self.position',self.position)
        return self.position, self.dist #means PSO result has collision,so ask for PathPlanning



    def RandomlyWalk(self):
        
        self.dist = self.EuclideanDistance(self.position,Goal)
        # print(f'Agent{self.ID} 119 curent position is {self.position} RandomlyWalk')
        if self.dist <= Epsilon:
            self.dist = -2
            return
        # if np.random.rand() < 0.5:
        # print(f'Agent{self.ID} 126 curent position is {self.position} RandomlyWalk')
        # AvailableCords = find_coordinates_inside_circle(self.position, self.sensorRadius, gridMap)
        AvailableCords = find_coordinates_inside_ring(self.position, self.max_vel, self.max_vel+1, gridMap)
        if AvailableCords:
            newPosition = np.array(random.choice(AvailableCords))
            # print(f'Agent{self.ID} 128 curent position is {self.position} RandomlyWalk')
            if no_obstacle_on_line(gridMap, self.position, newPosition) and gridMap[tuple(newPosition)] == 0:
                # velocity = self.VelocityLimit(newPosition - self.position, self.max_vel).astype(int)
                # self.position = self.position + velocity
                self.position = newPosition
                # print(f'Agent{self.ID} 130 curent position is {self.position} RandomlyWalk')
                self.dist = self.EuclideanDistance(self.position, Goal)
                if self.dist < self.pBestDis:
                    self.pBestDis = self.dist
                    self.pbestPos = self.position
                if self.dist < self.EuclideanDistance(self.Gbest, Goal):
                    self.Gbest = self.position
                if self.dist <= Epsilon:
                    self.dist = -2
                    # print(f'Agent{self.ID} 139 curent position is {self.position} RandomlyWalk')
                    return
                return
            else:
                print('228 choose failed')
                self.dist = self.EuclideanDistance(self.Gbest,Goal) #wait for the next time randomly walk
                return

        else:
            print(f'ID {self.ID} has no AvailableCords!')
            print('agent position is: ', self.position)
            self.dist = self.EuclideanDistance(self.Gbest,Goal) #wait for the next time randomly walk
            # print(f'Agent 149 curent position is {self.position} RandomlyWalk')
            # return

        return self.position, self.dist
    

    def AConnection(self,Gbest,dist):
        # if self.dist > Epsilon and dist == -2:
        #     print(f'agent ID {agent.id}, position: {agent.position}')
        # print(f'dist is {dist}')
        if self.dist == -2:
            return self.position,self.dist
        elif dist== -3 : # downlink/uplink failed,randomly walk mode
            # self.position,self.dist = self.RandomlyWalk()
            self.RandomlyWalk()
            # self.dist = -1 #ask for help, stay there
            return self.position,self.dist
            
        elif dist >= 0:
            # self.RandomlyWalk()
            # self.position, self.dist = self.PSOvelocity(Gbest,dist)
            # print(f'Agent193 Aconnection got {dist} from Server')
            self.PSOvelocity(Gbest,dist)
            return self.position, self.dist

        
        # else:
            # print(f'Agent196 Aconnection got {dist} from Server ')
            
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

    # @staticmethod
    # def VelocityLimit(a, max_vel):
    #     e = np.linalg.norm(a)
    #     a = a.astype(float) / e
    #     a *= max_vel
    #     return a