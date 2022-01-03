import numpy as np 
import matplotlib.pyplot as plt 
import math

gamma = 1
alpha = 0.5
epsilon = 0.1

rows = 4
columns = 12

#0=up, 1=down, 2=left, 3=right
actionSet = [0,1,2,3]

startState = [3,0]
endState = [3,11]

def moving(state, action):
	i,j = state
	newState = [i,j]
	reward = -1

	if action==0:
		newState = [max(i-1,0), j]
	elif action==1:
		newState = [min(i+1, rows-1), j]
	elif action==2:
		newState = [i, max(j-1,0)]
	elif action==3:
		newState = [i, min(j+1, columns-1)]

	if (action==1 and i==2 and 1<=j<=10) or (action==3 and state==startState):
		reward = -100
		newState = startState

	return newState, reward

#on-policy algorithm
def Sarsa(Estimate_StateActionValue):

	rewards = 0
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

		rewards += reward

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

	return rewards


def QLearning(Estimate_StateActionValue):
	rewards = 0

	#init S
	state = startState

	while True:
		if state==endState:
			break

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

		#get S', R
		nextState, reward = moving(state, action)

		rewards += reward

		#update rule
		Estimate_StateActionValue[state[0], state[1], action] += alpha*(reward + gamma*(np.max(Estimate_StateActionValue[nextState[0], nextState[1], :])) - Estimate_StateActionValue[state[0], state[1], action])

		#update state
		state = nextState
	return rewards

def ExpectedSarsa(Estimate_StateActionValue):

	rewards = 0
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

		rewards += reward

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
		target = 0
		QNext = Estimate_StateActionValue[nextState[0], nextState[1], :]
		maximizingActions = np.argwhere(QNext == np.max(QNext))
		for actions in actionSet:
			if actions in maximizingActions:
				target += ((1-epsilon)/len(maximizingActions) + epsilon/len(actionSet)) * Estimate_StateActionValue[nextState[0], nextState[1], actions]
			else:
				target += epsilon/len(actionSet) * Estimate_StateActionValue[nextState[0], nextState[1], actions]

		Estimate_StateActionValue[state[0], state[1], action] += alpha*(reward + gamma*target - Estimate_StateActionValue[state[0], state[1], action])

		#update state and action
		state = nextState
		action = nextAction

	return rewards



def figure1():
	episodes = 500
	runs = 50

	rewards_Sarsa = np.zeros(episodes)
	rewards_QLearning = np.zeros(episodes)

	Estimate_StateActionValue_Sarsa = np.zeros((rows, columns, len(actionSet)))
	Estimate_StateActionValue_Qlearning = np.zeros((rows, columns, len(actionSet)))

	for run in range(0,runs):
		for i in range(0, episodes):
			rewards_Sarsa[i] += Sarsa(Estimate_StateActionValue_Sarsa)
			rewards_QLearning[i] += QLearning(Estimate_StateActionValue_Qlearning)

	rewards_Sarsa /= runs
	rewards_QLearning /= runs

	plt.plot(np.arange(0,episodes), rewards_Sarsa, label='sarsa')
	plt.plot(np.arange(0,episodes), rewards_QLearning, label='QLearning')
	plt.legend()
	plt.show()

figure1()