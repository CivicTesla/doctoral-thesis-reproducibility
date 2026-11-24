import numpy as np 
from cv2 import cv2
from matplotlib import pyplot as plt
#%matplotlib inline



class Agent():
    def __init__(self, init_position, init_velocity, scan_radius=np.Inf,neighborUpper=50,max_vel=5):
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


            #print('i===========',i)
            # find the neighbors
            for j in range(i+1,len(env)):
                Adist = np.linalg.norm(object.position-env[j].position)
                #print('j!!!!!!!',j)
                if Adist < self.scan_radius:
                    object.neighbors.append(j)
                    env[j].neighbors.append(i)

                    
            # with neighbor number limit
            #print('original object.neighbors ',object.neighbors)
            if len(object.neighbors)>object.neighborUpper:
                neighborTemp = []
                for n in object.neighbors:
                    AdistTemp = np.linalg.norm(object.position-env[n].position)
                    neighborTemp.append(AdistTemp)
                neighZipped = zip(object.neighbors,neighborTemp)
                neighZipped = sorted(neighZipped,key=lambda x:x[1])#sorted by AdistTemp
                del neighZipped[object.neighborUpper:]
                object.neighbors, _ = zip(*neighZipped)
            #print('zipped object.neighbors ',object.neighbors)




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

    NUMBER_OF_AGENTS = 50
    scanRList = np.array([30,50,70,np.inf])
    #nUpperList = np.array([3,7,11,50])

    Sumloop = []
    SumArray = []


    for j in range(300):
        loopArray = []
        covArray = []


        for [k,scanR] in enumerate(scanRList):
            print('simulation for radius ',scanR)
        #for [k,nUpper] in enumerate(nUpperList):
        #    print('simulation for neighbor number ',nUpper)

            # Populating agents
            agents = [] 
            for i in range(NUMBER_OF_AGENTS):
                #random initial postion and random velocity
                agents.append(Agent(np.random.rand(2)*display.shape[0],
                                     np.random.rand(2) - 0.5,scanR))
                #agents.append(Agent(np.random.rand(2)*display.shape[0],
                #                    np.random.rand(2) - 0.5,neighborUpper=nUpper))
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
        
    
                print('Working loop ',(i+1)*len(agents))
                i+=1
            loopArray.append(np.array(loopNum))
            covArray.append(np.array(covNum))
        
        #save result
        # np.save("loopArray.npy",loopArray)
        # np.save("covArray.npy",covArray)

        Sumloop.append(loopArray)
        SumArray.append(covArray)
        
        # plot the result
        fig, axes = plt.subplots()
        axes.set_xlabel('Loop number')  # 横轴名称
        axes.set_ylabel('converge number')
        axes.set_title('simulation final result')  # 图形名称
        for i in range(len(scanRList)):
        #for i in range(len(nUpperList)):
            axes.plot(loopArray[i],covArray[i],linewidth=2)
        axes.legend(["scan_radius=30", "scan_radius=40","scan_radius=50","scan_radius=60","scan_radius=70"],loc=0)
        #axes.legend(["upperNeighbors=3","upperNeighbors=7", "upperNeighbors=11","upperNeighbors=50"],loc=0)

        filename = 'neighbors'+str(j)+'.jpg'
        
        plt.savefig(filename)
        #plt.show()
    #save result
    np.save("SumloopNeighborRadius.npy",Sumloop)
    np.save("SumArrayNeighborRadius.npy",SumArray)

    