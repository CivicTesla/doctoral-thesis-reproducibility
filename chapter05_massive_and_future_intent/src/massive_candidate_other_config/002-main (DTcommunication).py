import numpy as np 
import cv2
from matplotlib import pyplot as plt
#%matplotlib inline
from random import choice



class Agent():
    def __init__(self, init_position, init_velocity, scan_radius=30,neighborUpper=np.inf,max_vel=5):
        self.position = init_position # Numpy array [x, y]
        self.velocity = init_velocity  # Numpy array [x, y]
        self.scan_radius = scan_radius # Int
        self.max_vel = max_vel
        self.pBest = np.Inf # single's position history
        self.pbestPos = np.Inf
        self.neighbors = []
        self.gbestPos = []
        self.neighborUpper=neighborUpper
      
    


    def update(self, env,image): #env is agents
        
        #PSO parameter
        c1=c2 = 2

        #loop for finding best position of agend and group
        for [i,object] in enumerate(env): #here env = agents
            #for the 1st loop, self.position = the 1st agent's position(constant value)
            distance = self.position - object.position
            # #clear results in case neighbors duplicate append 
            # object.neighbors = []

            #print('i===========',i)
            # define neighbors with communication


            #Neighbor list for choice, each time 2
            # Neighborlist = []
            # lenE = len(env)
            # for m in range(lenE):
            #     for l in range(lenE):
            #         Neighborlist.append([m,lenE-l-1])
 
            # append the neighbor
            # comFlag = 1000
            # if i==0:
            #     for ab in range(1000):
            #         com = choice(Neighborlist)
            #         Neighborlist.remove(com)
            #         env[com[0]].neighbors.append(com[1])
            #         env[com[1]].neighbors.append(com[0])
            #         comFlag -= 4
            #         if comFlag <=0 or not Neighborlist: #if 1000 channel used or PickList run out
            #             break
           

            #Original Neighbor counts
            # for j in range(i+1,len(env)):
            #     Adist = np.linalg.norm(object.position-env[j].position)
            #     #print('j!!!!!!!',j)
            #     if Adist < self.scan_radius:
            #         object.neighbors.append(j)
            #         env[j].neighbors.append(i)

                    
            # with neighbor number limit
            #print('original object.neighbors ',object.neighbors)
            # if len(object.neighbors)>object.neighborUpper:
            #     neighborTemp = []
            #     for n in object.neighbors:
            #         AdistTemp = np.linalg.norm(object.position-env[n].position)
            #         neighborTemp.append(AdistTemp)
            #     neighZipped = zip(object.neighbors,neighborTemp)
            #     neighZipped = sorted(neighZipped,key=lambda x:x[1])#sorted by AdistTemp
            #     del neighZipped[object.neighborUpper:]
            #     object.neighbors, _ = zip(*neighZipped)
            # #print('zipped object.neighbors ',object.neighbors)

            # if len(object.neighbors)>object.neighborUpper:
            #     object.neighbors = object.neighbors[object.neighborUpper:]



            #find best global position
            gBest = np.linalg.norm(PicGoal-object.position)+np.random.normal()
            object.gbestPos = object.position
            for k in range(len(env)):
                disTest = np.linalg.norm(PicGoal - env[k].position)+np.random.normal()
                #print('k',k)
                if disTest < gBest:
                    gBest = disTest
                    object.gbestPos = env[k].position
   
            
            #find personal best position
            pBestVal = np.linalg.norm(PicGoal - object.position)
            if pBestVal < object.pBest:
                object.pBest = pBestVal
                object.pbestPos = object.position
             
            #update veloctiy and position
            object.velocity = object.velocity + c1 * np.random.rand()*(object.pbestPos-object.position) + c2 * np.random.rand() * (object.gbestPos - object.position)

            #limit velocity up to max velocity
            if np.linalg.norm(object.velocity) > object.max_vel:
                object.velocity /= np.linalg.norm(object.velocity)
                object.velocity *= object.max_vel

            #calculate position
            # Calculating new position
            new_position = object.position + object.velocity


            # Updating position
            #if np.linalg.norm(new_position) < np.linalg.norm(object.pbestPos):
            object.position = new_position

            #clear results in case neighbors duplicate append 
            object.neighbors = []

        numFlag = 0
        epsilon = 5

        for [i,agent] in enumerate(env):
            agent.draw(image)
            if np.linalg.norm(env[i].position-PicGoal) < epsilon:
            #if np.array_equal(env[i].position.astype(int),np.array(PicGoal)):
                numFlag +=1
            #cv2.waitKey(0)
        cv2.drawMarker(display, PicGoal, (0,0,255), markerType=1,thickness = 6)
        cv2.imshow('progress',display)
        #dynamic show 27ms pro time
        cv2.waitKey(27)
        display[:,:,:] = 255

        return numFlag



    def draw(self, image):
        #cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(image, (int(round(self.position[0])), int(round(self.position[1]))), 5, (90,30,60), -1, cv2.LINE_AA)



if __name__ == "__main__":
    display = np.ones((600,640,3), dtype=np.uint8) * 255 #画布的尺寸; 255表示白色,3表示彩色通道

    #NUMBER_OF_AGENTS=50
    # AgentsNum = np.linspace(100,200,10)
    AgentsNum = np.arange(190,210,10)
    #AgentsNum = [10]
    #scanRList = np.array([30,50,70,np.inf])
    #nUpperList = np.array([3,7,11,50])

    Sumloop = []
    SumArray = []

    BigAgents = []
    Bigloop = []

    

    for Agts in AgentsNum:
        NUMBER_OF_AGENTS = int(Agts)
        print(NUMBER_OF_AGENTS)
        print('==============================')
        
        loopArray = []
        covArray = []

        for j in range(5):#Monte Carlo Loop 
            # for [k,scanR] in enumerate(scanRList):
            #     print('simulation for radius ',scanR)
            # for [k,nUpper] in enumerate(nUpperList):
            #     print('simulation for neighbor number ',nUpper)
        
            # Populating agents================================================================================
            agents = [] 
            # same initial postion and same initial velocity
            #np.random.seed(1)
            iPosition = np.random.rand(NUMBER_OF_AGENTS,2)*display.shape[0]
            #np.random.seed(2)
            iVel= np.random.rand(NUMBER_OF_AGENTS,2) - 0.5
            for i in range(NUMBER_OF_AGENTS):
                agents.append(Agent(iPosition[i],iVel[i]))
            #red X position
            PicGoal = (400,300)    
    
            #idx for while loop
            i = 0
            
            loopNum = []
            covNum = []
                
    
            while i>=0:
                for [j,agent] in enumerate(agents):
                    #agent.draw(display)
                    covVal = agent.update(agents,display)
                    loopNum.append(j+i*len(agents))
                    covNum.append(covVal)
                if loopNum[-1]>=300:
                    break    
        
                # print('Loop i',i)
                # print('len(agents)',len(agents))
                #print('Working loop ',(i+1)*len(agents))
                i+=1
            loopArray.append(np.array(loopNum[:100])) #all the monteCarlo Loop result
            covArray.append(np.array(covNum[:100]))
            #loopArray.append(np.array(loopNum)) #all the monteCarlo Loop result
            #covArray.append(np.array(covNum))
            #backards tab===========================================================================================

        Sumloop.append(loopArray) #each append for each agent-group
        SumArray.append(covArray)
            
        
    #save result
    np.save("SumloopDT.npy",Sumloop)
    np.save("SumArrayDT.npy",SumArray)
    # BigAgents.append(SumArray)
    # Bigloop.append(Sumloop)
    # SumArray = []
    # Sumloop = []

    # np.save("BigAgents.npy",BigAgents)
    # np.save("Bigloop.npy",Bigloop)

