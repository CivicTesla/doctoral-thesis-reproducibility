import numpy as np 
#import cv2
import random
from LocalSensor import find_coordinates_inside_circle, find_coordinates_inside_ring
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
        self.sensorRadius = 6
        self.CurrentDist = self.EuclideanDistance(self.position,Goal)
        self.ID = agent_id
        self.dist = init_dist
        self.Gbest = self.position


    # def PSOvelocity(self,Gbest,dist): # using PSO to return position and distance
    #     # CurrentDist = self.EuclideanDistance(self.position, Goal)
    #     self.dist = self.EuclideanDistance(self.position,Goal)
    #     if self.dist <= Epsilon:
    #         self.dist = -2
    #         return

    #     if np.any(Gbest == self.position) and self.dist > Epsilon:
    #         return self.RandomlyWalk()
        
    #     if self.dist < self.pBestDis:
    #         self.pBestDis = self.dist
    #         self.pbestPos = self.position

    #     if self.dist < dist:
    #         self.gBestDis = self.dist
    #         self.Gbest = self.pbestPos = self.position
    #     else:
    #         self.Gbest = Gbest
        
    #     if dist < self.gBestDis:
    #         # self.gBestDis = dist #because when dist = -2, it could not be used to compare
    #         self.Gbest = Gbest
    #         # print('before position',self.position)
    #     # velocity = w * self.velocity + c1 * np.random.rand()*(self.pbestPos-self.position) + \
    #     #         c2 * np.random.rand() * (self.Gbest - self.position)+ np.random.normal(0,1)*self.max_vel
        
    #     #PSO parameter
    #     c1=c2=2
    #     # print('befor PSO agent position is ',self.position)
    #     w = 0.9 #w represents the global search capability, with a larger value indicating a stronger search ability."

    #     velocity = w * self.velocity + c1 * np.random.uniform(-1,1)*(self.pbestPos-self.position) + \
    #             c2 * np.random.uniform(-1,1) * (self.Gbest - self.position)+ np.random.uniform(-1,1)*2


    #     velocity = self.VelocityLimit(velocity,self.max_vel)
    #     # print('velocity',velocity)
    #     # position = self.position + velocity
    #     # print('before pso position: ',self.position)
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
    
    #     # if no_obstacle_on_line(gridMap, self.position,position):
    #         # print('no obstacle on line: ')
    #         # print('no no_obstacle_on_line',position)
    #     if is_valid_position and no_obstacle_on_line(gridMap, self.position, new_position):
    #         self.position = new_position
    #         self.velocity = velocity
    #         self.dist = self.EuclideanDistance(self.position,Goal)
    #         if self.dist < self.pBestDis:
    #             self.pBestDis = self.dist
    #             self.pbestPos = self.position
    #         # print('successful PSO agent positionis ',self.position)
    #         # print('successful pso new position',self.position)
    #         if 0<= self.dist <= Epsilon: #means achieved Goal:
    #             self.dist = -2 # means achieved Goal
    #             return self.position, self.dist
    #         return self.position,self.dist
    #     # print('failed PSO agent position is ',self.position)
    #     self.dist = -1
    #     # print('self.position after dist = -1',self.position)
    #     # print('pso self.position',self.position)
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
        self.dist = self.EuclideanDistance(self.position,Goal)
        if self.dist <= Epsilon:
            self.dist = -2
            return
        if not no_obstacle_on_line(gridMap,self.position,Step):#check if Step is an obstacle
            print(f'Sesame has error!!!,Step is {Step}')
            print(f'corresponding agent position is {self.position}')
            # self.dist = -1 #ask for a new path
            self.RandomlyWalk()#just walk away to find new land
            return

        velocity = Step - self.position

        # 检查新位置是否合法
        new_position = (self.position + velocity).astype(int)
        if gridMap[new_position[0], new_position[1]] != 1:
            # 通过速度限制
            if np.linalg.norm(velocity) > self.max_vel:
                velocity = self.VelocityLimit(velocity, self.max_vel).astype(int)
                newPosition = self.position + velocity
                if no_obstacle_on_line(gridMap,self.position,newPosition):
                    self.position = (self.position + velocity).astype(int)
            else:
                self.position = Step

            # 更新距离等信息
            self.dist = self.EuclideanDistance(self.position, Goal)
            if self.dist < self.pBestDis:
                self.pBestDis = self.dist
                self.pbestPos = self.position

            if self.dist <= Epsilon:
                self.dist = -2  # 达到目标
                self.Gbest = self.position  # 更新全局最优位置
        else:
            print("Collision detected at new position. Handling collision...")
            self.dist = -1  # 处理碰撞

        return self.position, self.dist

    # def RandomlyWalk(self):
    #     self.dist = self.EuclideanDistance(self.position,Goal)
    #     if self.dist <= Epsilon:
    #         self.dist = -2
    #         return
    #     if np.random.rand() < 0.5:
    #         AvailableCords = find_coordinates_inside_circle(self.position, self.sensorRadius, gridMap)
    #         if AvailableCords:
    #             newPosition = np.array(random.choice(AvailableCords))
    #             if no_obstacle_on_line(gridMap, self.position, newPosition):
    #                 velocity = self.VelocityLimit(newPosition - self.position, self.max_vel).astype(int)
    #                 self.position = self.position + velocity
    #                 self.dist = self.EuclideanDistance(self.position, Goal)
    #                 if self.dist < self.pBestDis:
    #                     self.pBestDis = self.dist
    #                     self.pbestPos = self.position
    #                 if self.dist < self.EuclideanDistance(self.Gbest, Goal):
    #                     self.Gbest = self.position
    #                 if self.dist <= Epsilon:
    #                     self.dist = -2
    #             else:
    #                 self.dist = -1 #ask for help

    #         else:
    #             print(f'ID {self.ID} has no AvailableCords!')
    #             print('agent position is: ', self.position)
    #             self.dist = -1  # ask for help, stay there

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
    #         else:
    #             self.dist = -1
 
    #     return self.position, self.dist

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
                # self.dist = self.EuclideanDistance(self.Gbest,Goal) #wait for the next time randomly walk
                self.dist = -1
                return
    
        else:
            print(f'ID {self.ID} has no AvailableCords!')
            print('agent position is: ', self.position)
            self.dist = self.EuclideanDistance(self.Gbest,Goal) #wait for the next time randomly walk
            # print(f'Agent 149 curent position is {self.position} RandomlyWalk')
            # return
    
        return self.position, self.dist



            

    def AConnection(self,Gbest,dist):
        # if dist == -2:
        #     return self.position,self.dist
        # if dist >= 0 or dist == -2:
        if dist >= 0:
            # self.position, self.dist = self.PSOvelocity(Gbest,dist)
            self.PSOvelocity(Gbest,dist)
            return
        # if self.dist > Epsilon and dist == -2:
        #     print(f'agent ID {agent.id}, position: {agent.position}')
        elif dist== -3 : # downlink/uplink failed,randomly walk mode
            # self.position,self.dist = self.RandomlyWalk()
            self.RandomlyWalk()
            return
            # self.dist = -1 #ask for help, stay there
            # return self.position,self.dist
        
        elif dist == -1: #-1 means use PathPlanning Mode
            # self.position, self.dist = self.SesameVelocity(Gbest)
            self.SesameVelocity(Gbest)
            return
        # else:
        #     print(f'got {dist} from Server')
            
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
