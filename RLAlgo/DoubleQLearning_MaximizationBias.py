import numpy as np 
import matplotlib.pyplot as plt 
import copy
import math

# A=0, B=1, Terminal state = 2
stateSet = [0,1,2]
startState = 0

#left = 0, right = 1
actionSetA = [0,1]
#action branches leaving state B to terminal state
actionSetB = [0,1,2,3,4,5,6,7,8,9]
actionSet = [actionSetA, actionSetB]

epsilon = 0.1
alpha = 0.1
gamma = 1


def moving(state, action):
	newState = 0
	reward = 0
	if state==0:
		if action==0:
			newState = 1
		elif action==1:
			newState = 2
		return newState, reward
	else:
		newState = 2
		reward = np.random.normal(-0.1,1)
		return newState, reward


def choose_action(state, q_value):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actionSet[state])
    else:
    	#np.argmax DOES NOT WORK HERE FOR DOUBLE Q LEARNING. DONT KNOW WHY BUT BREAKING TIES RANDOMLY SEEMS TO WORK HERE
        values_ = q_value[state]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        

def DoubleQLearning(Estimate_StateActionValue1, Estimate_StateActionValue2):
	leftCount = 0
	#init S
	state = startState

	while True:
		if state==2:
			break

		#choose epsilon-greedy action
		action = choose_action(state, [item1 + item2 for item1, item2 in zip(Estimate_StateActionValue1, Estimate_StateActionValue2)])

		#get S', R
		nextState, reward = moving(state, action)
		

		if state==0 and action==0:
			leftCount += 1

		#update rule
		if np.random.binomial(1,0.5)==1:
			#print(Estimate_StateActionValue1)
			#maximizingAction = np.random.choice([action_ for action_, value_ in enumerate(Estimate_StateActionValue1[nextState]) if value_ == np.max(Estimate_StateActionValue1[nextState])])			
			temp = Estimate_StateActionValue1[nextState].copy()
			maximizingAction = np.argmax(temp)
			Estimate_StateActionValue1[state][action] += alpha*(reward + gamma*(Estimate_StateActionValue2[nextState][maximizingAction]) - Estimate_StateActionValue1[state][action])
		else:
			#print(Estimate_StateActionValue2[nextState])
			temp = Estimate_StateActionValue2[nextState].copy()
			maximizingAction = np.argmax(temp)
			Estimate_StateActionValue2[state][action] += alpha*(reward + gamma*(Estimate_StateActionValue1[nextState][maximizingAction]) - Estimate_StateActionValue2[state][action])

		#update state
		state = nextState
	return leftCount

def QLearning(Estimate_StateActionValue):
	leftCount = 0
	#init S
	state = startState

	while True:
		if state==2:
			break

		#choose epsilon-greedy action
		action = choose_action(state, Estimate_StateActionValue)

		#get S', R
		nextState, reward = moving(state, action)
		

		if state==0 and action==0:
			leftCount += 1

		#update rule
		Estimate_StateActionValue[state][action] += alpha*(reward + gamma*(np.max(Estimate_StateActionValue[nextState])) - Estimate_StateActionValue[state][action])

		#update state
		state = nextState
	return leftCount


def figure():
	episodes = 300
	runs = 1000
	leftCount_DoubleQLearning = np.zeros(episodes)
	leftCount_QLearning = np.zeros(episodes)
	#action choice prob for state 0,1,2
	Estimate_StateActionValue = [np.zeros(2), np.zeros(len(actionSetB)), np.zeros(1)]

	for run in range(0, runs):
		Estimate_StateActionValue1 = copy.deepcopy(Estimate_StateActionValue)
		Estimate_StateActionValue2 = copy.deepcopy(Estimate_StateActionValue)
		Estimate_StateActionValue3 = copy.deepcopy(Estimate_StateActionValue)

		for i in range(0, episodes):
			leftCount_DoubleQLearning[i] += DoubleQLearning(Estimate_StateActionValue1, Estimate_StateActionValue2)
			leftCount_QLearning[i] += QLearning(Estimate_StateActionValue3)


	leftCount_DoubleQLearning /= runs
	leftCount_QLearning /= runs

	plt.plot(np.arange(0, episodes), leftCount_DoubleQLearning, label='DoubleQLearning')
	plt.plot(np.arange(0, episodes), leftCount_QLearning, label='QLearning')
	plt.legend()
	plt.show()

figure()