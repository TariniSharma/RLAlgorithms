import numpy as np
import matplotlib.pyplot as plt 

#Init
#keep global unchanging variables here
k = 10 #k arms
epsilon = 0
timesteps = 1000
runs = 2000 

def getTrueActionValues(True_ActionValue):
	samples = np.random.normal(0,1,k)
	for i in range(0,k):
		True_ActionValue[i] = samples[i]

def greedyActionSelection(Estimate_ActionValue):
	# wp epsilon select random action, wp 1-epsilon select greedy action
	if np.random.uniform(0,1)<epsilon:
		arm = np.random.randint(0,k,dtype='int')
	else:
		arm = np.argmax(Estimate_ActionValue)
	return arm

def getReward(arm,True_ActionValue):
	reward = np.random.normal(True_ActionValue[arm],1)
	return reward

def banditAlgorithm():
	#Initialization
	Reward = np.zeros(timesteps, dtype='float') #reward received at each timestep
	
	#Loop for 2000 runs
	for j in range(0,runs):
		#Init
		Count_Action = np.zeros(k)
		Estimate_ActionValue = np.zeros(k)
		True_ActionValue = np.zeros(k)
		getTrueActionValues(True_ActionValue)

		#Loop for 1000 timesteps
		for i in range(0,timesteps):
			GreedyAction = greedyActionSelection(Estimate_ActionValue)
			reward = getReward(GreedyAction,True_ActionValue)
			Count_Action[GreedyAction] += 1
			Estimate_ActionValue[GreedyAction] += (reward - Estimate_ActionValue[GreedyAction])/Count_Action[GreedyAction]
			Reward[i] += reward

	Reward /= runs

	#Plotting
	time = [i for i in range(0,timesteps)]
	plt.figure()
	plt.plot(time, Reward, label='∈ = 0.1')
	plt.xlabel('steps')
	plt.ylabel('average reward')
	plt.legend()
	plt.show()

banditAlgorithm()