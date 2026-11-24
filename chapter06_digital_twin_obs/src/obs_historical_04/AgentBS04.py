import numpy as np 
import cv2
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
        self.getFlag = False


    def PSOvelocity(self,Gbest,dist): # using PSO to return position and distance
        # CurrentDist = self.EuclideanDistance(self.position, Goal)
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return
        if np.any(self.Gbest == self.position):
            # print(f'Agent ID {agent.ID}nice ~~~~~~~~~~~~')
            if self.dist == -2:
                return
            return self.RandomlyWalk()
        #PSO parameter
        c1=c2=2
        # print('befor PSO agent position is ',self.position)
        w = 0.9 #w represents the global search capability, with a larger value indicating a stronger search ability."
        if self.dist < self.pBestDis:
            self.Gbest = self.position

        if self.dist < dist:
            self.gBestDis = self.dist
            self.Gbest = self.position
        else:
            self.Gbest = Gbest
        if self.dist < self.pBestDis:
            self.pBestDis = self.dist
            self.pbestPos = self.position
        # if dist < self.gBestDis:
        #     self.gBestDis = dist
            # print('before position',self.position)
        # velocity = w * self.velocity + c1 * np.random.rand()*(self.pbestPos-self.position) + \
        #         c2 * np.random.rand() * (self.Gbest - self.position)+ np.random.normal(0,1)*self.max_vel
        

        velocity = w * self.velocity + c1 * np.random.uniform(-1,1)*(self.pbestPos-self.position) + \
                c2 * np.random.uniform(-1,1) * (self.Gbest - self.position)+ np.random.uniform(-1,1)* 2


        velocity = self.VelocityLimit(velocity,self.max_vel)
        # print('velocity',velocity)
        position = self.position + velocity
        # print('before pso position: ',self.position)
        # print('after pso position:',position)
        
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
        else:
            # print(f'Agent {self.ID},Pso 2 RandomlyWalk, position {self.position} ')
            self.RandomlyWalk()
        # self.dist = self.EuclideanDistance(self.position,Goal)
        # return self.position, self.dist #means PSO result has collision,so ask for PathPlanning

    # def RandomlyWalk(self):
    #     if np.random.rand() < 0.5:
    #         AvailableCords = find_coordinates_inside_circle(self.position, self.sensorRadius, gridMap)
    #         while True:
    #             if not AvailableCords:#means no Available Cords found
    #                 newPosition = None
    #                 break
    #             if AvailableCords:
    #                 # Randomly choose a coordinate from AvailableCords
    #                 newPosition = np.array(random.choice(AvailableCords))
    #                 if no_obstacle_on_line(gridMap,self.position,newPosition):
    #                     break
    #                     # Remove the chosen coordinate from AvailableCords
    #             AvailableCords.remove(newPosition)   
                
    #         if newPosition:
    #             self.position = newPosition
    #             self.dist = self.EuclideanDistance(self.position,Goal)
            

    #     else:
    #         velocity = self.VelocityLimit(self.Gbest - self.position,self.max_vel).astype(int)
    #         newPosition = self.position + velocity
    #         if no_obstacle_on_line(gridMap, self.position, newPosition):
    #             self.position = newPosition
    #             self.velocity = velocity
    #             self.dist = self.EuclideanDistance(self.position, Goal)
    #             if self.dist < self.pBestDis:
    #                 self.pBestDis = self.dist
    #                 self.pbestPos = self.position
    #             if self.dist < self.EuclideanDistance(self.Gbest, Goal):
    #                 self.Gbest = self.position
    #             if self.dist <= Epsilon:
    #                     self.dist = -2
    #         self.dist = self.EuclideanDistance(self.position,Goal)
 
    #     return self.position, self.dist



    def RandomlyWalk(self):
        # print(f'before RandomlyWalk {self.ID}, Agent position is {self.position}')
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return


        if np.random.rand() < 0.5:
            
            AvailableCords = find_coordinates_inside_circle(self.position, self.sensorRadius, gridMap)
            while True:
                if not AvailableCords:#means no Available Cords found
                    newPosition = None
                    break
                if AvailableCords:
                    # Randomly choose a coordinate from AvailableCords
                    newPosition = np.array(random.choice(AvailableCords))
                    if no_obstacle_on_line(gridMap,self.position,newPosition):
                        # print('newPosition is ',newPosition)
                        break
                        # Remove the chosen coordinate from AvailableCords
                AvailableCords.remove(newPosition)   
                
            if np.any(newPosition):
                self.position = newPosition
                self.dist = self.EuclideanDistance(self.position,Goal)
                if self.dist <= Epsilon:
                        self.dist = -2
        else:
            velocity = self.VelocityLimit(self.Gbest - self.position,self.max_vel).astype(int)
            newPosition = self.position + velocity
            if no_obstacle_on_line(gridMap, self.position, newPosition):
                self.position = newPosition
                self.velocity = velocity
                self.dist = self.EuclideanDistance(self.position, Goal)
                if self.dist < self.pBestDis:
                    self.pBestDis = self.dist
                    self.pbestPos = self.position
                if self.dist < self.EuclideanDistance(self.Gbest, Goal):
                    self.Gbest = self.position
                if self.dist <= Epsilon:
                        self.dist = -2
        # print(f'after RandomlyWalk {self.ID}, Agent position is {self.position}')
        return self.position, self.dist



            

    def AConnection(self,Gbest,dist):
        # if self.dist > Epsilon and dist == -2:
        #     print(f'agent ID {agent.id}, position: {agent.position}')
        if dist== -3 : # downlink/uplink failed,randomly walk mode
            # self.position,self.dist = self.RandomlyWalk()
            self.RandomlyWalk()
                

            # self.dist = -1 #ask for help, stay there
            # return self.position,self.dist
            
        if dist >= 0:
            # self.position, self.dist = self.PSOvelocity(Gbest,dist)
            self.PSOvelocity(Gbest,dist)
        
        # elif dist == -1: #-1 means use PathPlanning Mode
        #     # self.position, self.dist = self.SesameVelocity(Gbest)
        #     self.SesameVelocity(Gbest)
            
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
