import numpy as np
import matplotlib.pyplot as plt 

#Init
gridRows = 4
gridColumns = 4
actionSet = [0,1,2,3] 
#0 - up, 1 - down, 2 - right, 3 - left
#pi is to choose each action with equal prob
threshold = 1e-4
gamma = 1
terminalStates = [[0,0],[3,3]]

def step(currentState, action):
	i,j = currentState
	reward = 0
	newState = [0,0]
	if(currentState in terminalStates):
		return newState, reward
	if action==0:
		reward = -1
		if i==0:
			newState[0] = 0
			newState[1] = j
		else:
			newState[0] = i-1
			newState[1] = j
	elif action==1:
		reward = -1
		if i==gridRows-1:
			newState[0] = i
			newState[1] = j
		else:
			newState[0] = i+1
			newState[1] = j
	elif action==2:
		reward = -1
		if j==gridColumns-1:
			newState[0] = i
			newState[1] = j
		else:
			newState[0] = i
			newState[1] = j+1
	elif action==3:
		reward = -1
		if j==0:
			newState[0] = i
			newState[1] = j
		else:
			newState[0] = i
			newState[1] = j-1
	#if newState in terminalStates:
	#	reward = 0 #incorrect
	return newState, reward


def valueIteration():
	Estimate_StateValue = np.zeros((gridRows, gridColumns))
	pi = np.zeros((gridRows,gridColumns,len(actionSet)))
	pi.fill(0.25)
	#policy evaluation
	while True:
		delta = 0
		for i in range(0,gridRows):
			for j in range(0,gridColumns):
				oldEstimate_StateValue = Estimate_StateValue[i,j]
				val = [0 for k in range(0,len(actionSet))]
				for action in actionSet:
					newState, reward = step([i,j], action)
					i1,j1 = newState
					val[action] = 1*(reward+gamma*Estimate_StateValue[i1,j1])
				Estimate_StateValue[i,j] = np.max(val)
				delta = max(delta, abs(oldEstimate_StateValue - Estimate_StateValue[i,j]))
		if(delta<threshold):
			break
	print("estimate:")
	print(Estimate_StateValue)	

	#policy improvement
	for i in range(0,gridRows):
		for j in range(0,gridColumns):
			val = [0 for k in range(0,len(actionSet))]
			for action in actionSet:
				newState, reward = step([i,j], action)
				i1,j1 = newState
				val[action] = 1*(reward+gamma*Estimate_StateValue[i1,j1])
			newActions = []
			maxval = np.max(val)
			for action in actionSet:
				if val[action] == maxval:
					newActions.append(action)
			for action in actionSet:
				if action in newActions:
					pi[i,j,action] = 1/len(newActions)
				else:
					pi[i,j,action] = 0	
	print("policy:")
	for i in range(0,gridRows):
		for j in range(0,gridColumns):
			print('[',end='')
			for action in actionSet:
				if pi[i,j,action]!=0:
					print(action,end=',')
			print(']',end=' ')
		print()
		
valueIteration()
#should not update estimate if terminal state