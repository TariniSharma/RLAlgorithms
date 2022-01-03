import numpy as np
import matplotlib.pyplot as plt

#actions - left=0, right=1
actionSet = [0,1]

#target policy
def targetPolicy():
	return 0

#behaviour policy
def behaviourPolicy():
	if np.random.binomial(1, 0.5) == 1:
	       return 1
	return 0

def OffPolicyPrediction(numEpisodes):
	#initialize rhos and returns from the behaviour policy for all episodes
	rhos = np.zeros(numEpisodes)
	behaviourPolicy_returns = np.zeros(numEpisodes)

	for iteration in range(0,numEpisodes):
		returns, episode = play(behaviourPolicy)
		# rho = 1
		numerator = 1.0
		denominator = 1.0
		for action in episode:
			if action == targetPolicy():
				denominator *= 0.5
			else:
				numerator = 0.0
				break
		rho = numerator/denominator
		rhos[iteration] = rho
		behaviourPolicy_returns[iteration] = returns

	#weightedReturns = targetPolicy_returns
	weightedReturns = rhos * behaviourPolicy_returns

	weightedReturns = np.add.accumulate(weightedReturns)
	rhos = np.add.accumulate(rhos)

	ordinarySampling = weightedReturns / np.arange(1, numEpisodes+1)

	return ordinarySampling

def play(policy):
	episode = []

	while True:
		action = policy()
		episode.append(action)
		if action == 1:
			return 0, episode
		if np.random.binomial(1,0.9) == 0:
			return 1, episode


def fig5_4():
	numEpisodes = 50000
	runs = 10
	for i in range(0,runs):
		ordinarySampling = OffPolicyPrediction(numEpisodes)
		plt.plot(np.arange(1,numEpisodes+1),ordinarySampling, label='run'+str(i+1))
	plt.xscale('log')
	plt.legend()
	plt.show()

fig5_4()