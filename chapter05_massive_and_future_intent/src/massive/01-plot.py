import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

# sumCov = np.load("02-SumArrayNeighbor.npy")
# sumLoop = np.load("02-SumloopNeighbor.npy")

sumCov = np.load("04-SumArrayRandomNeighbors.npy")
sumLoop = np.load("04-SumloopRandomNeighbors.npy")


# print(sumCov.shape)
# print(sumLoop.shape)
# print(sumLoop[0][0])
Cov = []
Loop = []

for i in range(sumCov.shape[1]):
    sumC = []
    sumL = []
    for j in range(sumCov.shape[0]):
        sumC.append(sumCov[j][i])
        sumL.append(sumLoop[j][i])
    Cov.append(sumC)
    Loop.append(sumL)

CovMean = []
# test = np.mean(Cov[0],axis=0)
# fig, axes = plt.subplots()
# axes.plot(test,linewidth=2)
# plt.show()
# print(len(test))
for k in range(sumCov.shape[1]):
    CovMean.append(np.mean(Cov[k],axis = 0))
        

# plot the result
fig, axes = plt.subplots()
axes.set_xlabel('Time in rounds')  # x-axis name
axes.set_ylabel('Number of convergend agents')
#axes.set_title('50 agents in total')  # graph name
#for i in range(len(scanRList)):
for i in range(sumCov.shape[1]):
    axes.plot(CovMean[i],linewidth=1)
    axes.set_yscale('log')
#axes.legend(["scan_radius=30", "scan_radius=50","scan_radius=70","scan_radius=Inf"],loc=0)
# axes.legend(["upperNeighbors=3","upperNeighbors=5", "upperNeighbors=7","upperNeighbors=11"],loc=0)
axes.legend(["$K=3$","$K=5$", "$K=7$","$K=1$1"],loc=0)
#plt.grid(True)
# plt.savefig('02-Neighbor.jpg')
#plt.savefig('02-GridonRandomNeighbor.png',dpi=800)
plt.savefig('02-RandomNeighbor.png',dpi=800)
plt.show()
