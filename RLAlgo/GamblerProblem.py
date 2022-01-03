import numpy as np
import matplotlib.pyplot as plt 

#Init
#stateSet = money that gambler has at any time of placing bet = [1..100]
#1D state here
#here actionSet is dependent on the state you are in(depending on the amt of money
#gambler has, he/she can place appropriate bets not exceeding the money it has)
gridRows = 100
threshold = 1e-9
gamma = 1
stateSet = np.arange(gridRows+1)

def valueIteration():
	Estimate_StateValue = np.zeros(gridRows+1)
	#IMP-1
	Estimate_StateValue[gridRows]=1
	pi = np.zeros(gridRows+1)
	list_EstimateStateValue = []
	#policy evaluation
	while True:
		delta = 0
		temp1 = Estimate_StateValue.copy()
		list_EstimateStateValue.append(temp1)
		for i in stateSet[1:]:
			oldEstimate_StateValue = Estimate_StateValue[i]
			val = []
			for action in range(0,min(i,gridRows-i)+1):
				temp = 0.4*(gamma*Estimate_StateValue[i+action]) + 0.6*(gamma*Estimate_StateValue[i-action])
				val.append(temp)
			Estimate_StateValue[i] = np.max(val)
			delta = max(delta, abs(oldEstimate_StateValue - Estimate_StateValue[i]))
		if(delta<threshold):
			temp1 = Estimate_StateValue.copy()
			list_EstimateStateValue.append(temp1)
			break
	print("estimate:")
	print(Estimate_StateValue)	

	#policy improvement
	for i in stateSet[1:gridRows]:
		val = []
		for action in range(0,min(i,gridRows-i)+1):
			temp = 0.4*(gamma*Estimate_StateValue[i+action]) + 0.6*(gamma*Estimate_StateValue[i-action])
			val.append(temp)
		#IMP-2	
		pi[i] = np.argmax(np.round(val[1:], 5)) +1
	print("policy:")
	print(pi)

	plt.figure(figsize=(10,20))
	plt.subplot(2,1,1)
	i=1
	for elm in list_EstimateStateValue:
		plt.plot(elm,label='sweep '+str(i))
		i += 1
	plt.xlabel('capital')
	plt.ylabel('value estimates')
	plt.legend()
	#plt.show()

	plt.subplot(2,1,2)
	plt.plot(pi)
	plt.xlabel('capital')
	plt.ylabel('value estimates')
	plt.show()

		
valueIteration()
#should not update estimate if terminal state