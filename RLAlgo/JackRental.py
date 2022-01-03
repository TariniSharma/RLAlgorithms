import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import poisson

#Init
#States = [i,j]
#i = num of cars in location 1 (0<=i<=20)
#j = num of cars in location 2 (0<=j<=20)
#Actions = num of cars moved from location 1 to 2 (-> +ve, <- -ve)
carsMoved = 5 #max 5 cars can be moved from 1 location to other
gridRows = 21 #[0,20] cars at any location
gridColumns = 21
actionSet = np.arange(-carsMoved, carsMoved+1)
movecarcost = 2
rentcarreward = 10
maxRental_RequestAndReturn = 11 #?

pi = np.zeros((gridRows,gridColumns))
#pi -> for each i and j, pi[i,j] represents the optimal action from actionset
#pi[i,j] gives us the optimal action to be selected in state [i,j]
#deterministic policy here. thats why did not use 3d matrix here. infact
#can do this for all policy iteration codes if we are only concerned with
#finding 1 optimal action for every state(instead of possible optimal actions)
#we are only concerned about 1 optimal action here
pi.fill(0.0)
threshold = 1e-1
gamma = 0.9

def step(currentState, action,Estimate_StateValue):
	#[i,j] are the current number of cars in both locations, move from 
	#1 location to other according to action. Then we open shop and
	#have requests for cars acccording to poisson and we get suitable 
	#rewards. Then we have returns for cars according to 
	#must loop for all possible requests and returns

	i,j = currentState
	returns = 0
	newState = [min(i-action,20),min(j+action,20)]
	returns -= movecarcost * abs(action)

	for rentalRequest_Location1 in range(maxRental_RequestAndReturn):
		for rentalRequest_Location2 in range(maxRental_RequestAndReturn):
			probRentalRequest_Location1 = poisson.pmf(rentalRequest_Location1,3)
			probRentalRequest_Location2 = poisson.pmf(rentalRequest_Location2,4)
			probRentalRequests = probRentalRequest_Location1 * probRentalRequest_Location2

			validRentalRequest_Location1 = min(rentalRequest_Location1, newState[0])
			validRentalRequest_Location2 = min(rentalRequest_Location2, newState[1])
			totalValidRentalRequests = validRentalRequest_Location1 + validRentalRequest_Location2
			reward = rentcarreward * (totalValidRentalRequests)

			#cars left at location 1 and 2 after valid rents done
			carsAtLocation1 = newState[0] - validRentalRequest_Location1
			carsAtLocation2 = newState[1] - validRentalRequest_Location2

			for rentalReturn_Location1 in range(maxRental_RequestAndReturn):
				for rentalReturn_Location2 in range(maxRental_RequestAndReturn):
					probRentalReturn_Location1 = poisson.pmf(rentalReturn_Location1,3)
					probRentalReturn_Location2 = poisson.pmf(rentalReturn_Location2,2)
					probRentalReturns = probRentalReturn_Location1 * probRentalReturn_Location2
					
					validRentalReturn_Location1 = min(rentalReturn_Location1, 20)
					validRentalReturn_Location2 = min(rentalReturn_Location2, 20)
			
					#cars at location 1 and 2 after valid returns done
					carsAtLocation1 = min(carsAtLocation1+validRentalReturn_Location1,20)
					carsAtLocation2 = min(carsAtLocation2+validRentalReturn_Location2,20)

					probRentalRequestsAndReturns = probRentalRequests * probRentalReturns

					returns += probRentalRequestsAndReturns * (reward + gamma*Estimate_StateValue[int(carsAtLocation1),int(carsAtLocation2)])

	
	return returns


def policyIteration():
	Estimate_StateValue = np.zeros((gridRows, gridColumns))
	while True:
		stop = False
		#policy evaluation
		while True:
			delta = 0
			for i in range(0,gridRows):
				for j in range(0,gridColumns):
					oldEstimate_StateValue = Estimate_StateValue[i,j]
					action = pi[i,j]
					returns = step([i,j], action, Estimate_StateValue)
					Estimate_StateValue[i,j] = returns
					delta = max(delta, abs(oldEstimate_StateValue - Estimate_StateValue[i,j]))
			if(delta<threshold):
				break
		print("estimate:")
		print(Estimate_StateValue)
		#policy improvement
		policyStable = True
		while True:
			for i in range(0,gridRows):
				for j in range(0,gridColumns):
					oldAction = pi[i,j]
					val = [0 for k in range(0,len(actionSet))]
					for action in actionSet:
						#check if valid action
						if (0<=action<=i) or (-j<=action<=0):
							val[action] = step([i,j], action, Estimate_StateValue)
						else:
							val[action] = -np.inf
					newAction = actionSet[argmax(val)]
					pi[i,j] = newAction
					if newAction != oldAction:
						policyStable = False
			if policyStable==True:
				stop=True
				break
		print("policy:")
		for i in range(0,gridRows):
			for j in range(0,gridColumns):
				print('[',end='')
				for action in actionSet:
					if pi[i,j,action]!=0:
						print(action,end=',')
				print(']',end=' ')
			print()

		if stop==True:
			break

policyIteration()
#should not update estimate if terminal state