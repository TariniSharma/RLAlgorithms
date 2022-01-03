import numpy as np 
import matplotlib.pyplot as plt 
import math as math

alpha = 0.5
epsilon = 0.1
gamma = 1

rows = 7
columns = 10
wind = [0,0,0,1,1,1,2,2,1,0]
Estimate_StateActionValue = np.zeros((rows, columns))

#0-up, 1-down, 2-left, 3-right
actionSet = [0,1,2,3]

#States - [i,j]
startState = [3,0]
endState = [3,7]

def moving(state, action):
	i,j = state
	newState = [i,j]
	reward = -1

	if action==0:
		newState = [max(0, i-1-wind[j]), j]
	elif action==1:
		newState = [max(min(rows-1, i+1-wind[j]),0), j]
	elif action==2:
		newState = [max(i-wind[j],0), max(j-1,0)]
	elif action==3:
		newState = [max(i-wind[j],0), min(j+1, columns-1)]

	return newState, reward

#on-policy algorithm
def Sarsa(Estimate_StateActionValue):

	timestep = 0
	#init S
	state = startState

	#choose epsilon-greedy action
	if np.random.binomial(1, epsilon) == 1:
		action = np.random.choice(actionSet)
	else:
		#choose maximising action a from Q(state,a)
		QFunForEachAction = np.zeros(len(actionSet))
		maxQFun = -math.inf
		for i in range(0, len(actionSet)):
			QFunForEachAction[i] = Estimate_StateActionValue[state[0]][state[1]][i]
			maxQFun = max(maxQFun, QFunForEachAction[i])
		action = np.argmax(QFunForEachAction)

	while True:
		if state==endState:
			break

		#get S', R
		nextState, reward = moving(state, action)

		#get A'
		#choose epsilon-greedy action
		if np.random.binomial(1, epsilon) == 1:
			nextAction = np.random.choice(actionSet)
		else:
			#choose maximising action a from Q(state,a)
			QFunForEachAction = np.zeros(len(actionSet))
			maxQFun = -math.inf
			for i in range(0, len(actionSet)):
				QFunForEachAction[i] = Estimate_StateActionValue[nextState[0]][nextState[1]][i]
				maxQFun = max(maxQFun, QFunForEachAction[i])
			nextAction = np.argmax(QFunForEachAction)

		#update rule
		Estimate_StateActionValue[state[0], state[1], action] += alpha*(reward + gamma*(Estimate_StateActionValue[nextState[0],nextState[1],nextAction] - Estimate_StateActionValue[state[0], state[1], action]))

		#update state and action
		state = nextState
		action = nextAction

		timestep+= 1
	return timestep

def figure():
	Estimate_StateActionValue = np.zeros((rows, columns, len(actionSet)))

	numEpisodes = 170
	timesteps = []
	for i in range(0, numEpisodes):
		timestep = Sarsa(Estimate_StateActionValue)
		timesteps.append(timestep)

	timesteps = np.add.accumulate(timesteps)

	plt.plot(timesteps, np.arange(1, numEpisodes+1))
	plt.show()

	#optimal policy = argmax a Q(S,a)
	optimalPolicy = np.zeros((rows,columns))
	for i in range(0, rows):
		for j in range(0, columns):
			#choose maximising action a from Q(state,a)
			QFunForEachAction = np.zeros(len(actionSet))
			maxQFun = -math.inf
			for a in range(0, len(actionSet)):
				QFunForEachAction[a] = Estimate_StateActionValue[i][j][a]
				maxQFun = max(maxQFun, QFunForEachAction[a])
			maxAction = np.argmax(QFunForEachAction)
			optimalPolicy[i,j] = maxAction

	print(optimalPolicy)

figure()
