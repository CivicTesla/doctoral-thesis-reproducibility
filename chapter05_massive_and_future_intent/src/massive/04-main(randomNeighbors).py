import numpy as np 
from cv2 import cv2
from matplotlib import pyplot as plt
#%matplotlib inline
from random import choice


class Agent():
    def __init__(self, init_position, init_velocity, neighborUpper,scan_radius=30,max_vel=5):
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


            #random choose neighbors without replicated or itself
            #while(len(object.neighbors)<object.neighborUpper):
            for iUp in range(object.neighborUpper):
            #for 
                nei = choice(range(len(env)))
                #if nei != i and nei not in object.neighbors and len(env[nei].neighbors)<object.neighborUpper:
                if nei != i and nei not in object.neighbors:
                    env[i].neighbors.append(nei)
                    env[nei].neighbors.append(i)
            #print('len(object.neighbors)',len(object.neighbors))
            #print('object.neighbors',object.neighbors)

            #find best group position
            gBest = np.linalg.norm(PicGoal-object.position)
            object.gbestPos = object.position
            for k in object.neighbors:
                disTest = np.linalg.norm(PicGoal - env[k].position)
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
        epsilon = 10

        for [i,agent] in enumerate(env):
            agent.draw(image)
            if np.linalg.norm(env[i].position-PicGoal) < epsilon:
            #if np.array_equal(env[i].position.astype(int),np.array(PicGoal)):
                numFlag +=1
            #cv2.waitKey(0)
        cv2.drawMarker(display, PicGoal, (0,0,255), markerType=1,thickness = 6)
        #cv2.imshow('progress',display)
        #dynamic show 27ms pro time
        cv2.waitKey(27)
        display[:,:,:] = 255

        return numFlag



    def draw(self, image):
        #cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(image, (round(self.position[0]), round(self.position[1])), 5, (90,30,60), -1, cv2.LINE_AA)

if __name__ == "__main__":
    display = np.ones((600,640,3), dtype=np.uint8) * 255 #画布的尺寸; 255表示白色,3表示彩色通道

    NUMBER_OF_AGENTS = 50
    #scanRList = np.array([30,40,50,60,70])
    nUpperList = np.array([3,5,7,11])
    
    Sumloop = []
    SumArray = []

    for j in range(300):
        loopArray = []
        covArray = []

        for [k,nUpper] in enumerate(nUpperList):
            print('simulation for neighbor number ',nUpper)
            # Populating agents
            agents = [] 
            for i in range(NUMBER_OF_AGENTS):
                #random initial postion and random velocity
                # agents.append(Agent(np.random.rand(2)*display.shape[0], 
                #                     np.random.rand(2) - 0.5,scanR))    
                agents.append(Agent(np.random.rand(2)*display.shape[0], 
                                    np.random.rand(2) - 0.5,neighborUpper=nUpper)) 
            #red X position
            PicGoal = (400,300)    
    
            #idx for while loop
            i = 0
            
            loopNum = []
            covNum = []
                
    
            while i>=0:
                for [f,agent] in enumerate(agents):
                    covVal = agent.update(agents,display)
                    loopNum.append(f+i*len(agents))
                    covNum.append(covVal)
                
                if loopNum[-1]>=250:
                    break    
        
    
                #print('Working loop ',(i+1)*len(agents))
                i+=1
            loopArray.append(np.array(loopNum))
            covArray.append(np.array(covNum))
        Sumloop.append(loopArray)
        SumArray.append(covArray)    
    #save result
    np.save("SumloopRandomNeighbors.npy",Sumloop)
    np.save("SumArrayRandomNeighbors.npy",SumArray)