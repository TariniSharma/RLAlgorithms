import numpy as np
import matplotlib.pyplot as plt 

#Init
#WORK IN PROGRESS!

gridRows = 4
gridColumns = 4
Estimate_StateValue = np.zeros((gridRows, gridColumns))
actionSet = [0,1,2,3] 
#0 - up, 1 - down, 2 - right, 3 - left
pi = np.zeros((gridRows,gridColumns))
pi.fill(0)
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
	return newState, reward


def policyIteration():
	while True:
		stop = False
		#policy evaluation
		while True:
			delta = 0
			for i in range(0,gridRows):
				for j in range(0,gridColumns):
					oldEstimate_StateValue = Estimate_StateValue[i,j]
					val = 0
					action = pi[i,j]
					newState, reward = step([i,j], action)
					i1,j1 = newState
					val = (reward+gamma*Estimate_StateValue[i1,j1])
					Estimate_StateValue[i,j] = val
					delta = max(delta, abs(oldEstimate_StateValue - Estimate_StateValue[i,j]))
			if(delta<threshold):
				break
		print("estimate:")
		print(Estimate_StateValue)
		#policy improvement
		policyStable = True
		for i in range(0,gridRows):
			for j in range(0,gridColumns):
				oldAction = pi[i,j]
				
				#oldActions = pi[i,j]
				val = [0 for k in range(0,len(actionSet))]
				newAction = 0
				for action in actionSet:
					newState, reward = step([i,j], action)
					i1,j1 = newState
					val[action] = reward + gamma*Estimate_StateValue[i1,j1]
				newAction = np.argmax(val)
				
				pi[i,j] = newAction
				if oldAction != newAction:
					policyStable = False
				#if newActions not in oldActions:
				#policyStable = False
		if policyStable==True:
			stop=True
		print("policy:")
		for i in range(0,gridRows):
			for j in range(0,gridColumns):
				print('[',end='')
				print(pi[i,j],end=',')
				print(']',end=' ')
			print()

		if stop==True:
			break

policyIteration()
#should not update estimate if terminal state