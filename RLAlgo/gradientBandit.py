import numpy as np
import matplotlib.pyplot as plt 

#Init
#keep global unchanging variables here
k = 10 #k arms
alpha = 0.1
timesteps = 1000
runs = 2000 

def getTrueActionValues(True_ActionValue):
	samples = np.random.normal(0,1,k)
	for i in range(0,k):
		True_ActionValue[i] = samples[i]

def gradientBanditActionSelection(Pit):
	arm = np.random.choice(np.arange(0,k),p=Pit)
	return arm

def getReward(arm,True_ActionValue):
	reward = np.random.normal(True_ActionValue[arm],1)
	return reward

def updatePit(Pit,Ht):
	expHt = np.exp(Ht)
	Pit = expHt/np.sum(expHt)
	return Pit

def banditAlgorithm():
	#Initialization
	Reward = np.zeros(timesteps, dtype='float') #reward received at each timestep
	Percentage_OptimalAction = np.zeros(timesteps,dtype='float')
	#Loop for 2000 runs
	for j in range(0,runs):
		#Init
		True_ActionValue = np.zeros(k)
		Ht = np.zeros(k)
		Pit = np.zeros(k)
		getTrueActionValues(True_ActionValue)
		baselineSum = 0
		baseline = 0

		#Loop for 1000 timesteps
		for i in range(0,timesteps):
			Pit = updatePit(Pit,Ht)
			GreedyAction = gradientBanditActionSelection(Pit)
			reward = getReward(GreedyAction,True_ActionValue)
			baselineSum += reward
			baseline = baselineSum/(i+1)
			for arm in range(0,k):
				if arm == GreedyAction:
					Ht[arm] = Ht[arm] + alpha*(reward-baseline)*(1-Pit[arm])
				else:
					Ht[arm] = Ht[arm] - alpha*(reward-baseline)*Pit[arm]
			Reward[i] += reward
			optAction = np.argmax(True_ActionValue)
			if(GreedyAction == optAction):
				Percentage_OptimalAction[i] += 1

	Reward /= runs
	Percentage_OptimalAction /= runs
	Percentage_OptimalAction *= 100
	#Plotting
	time = [i for i in range(0,timesteps)]
	plt.figure()
	plt.plot(time, Percentage_OptimalAction, label='∈ = 0.1')
	plt.xlabel('steps')
	plt.ylabel('Percentage of OptimalAction')
	plt.legend()
	plt.show()

banditAlgorithm()