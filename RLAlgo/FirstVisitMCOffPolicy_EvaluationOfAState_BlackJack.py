import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns

#states : playerSum(12-21), dealerFaceUpCard(Ace-10), usabilityAce(1-usable.0-unusable)

#Action: hit=0, sticks=1
actionSet = [0,1]

#policy: from dealerSum to action
#dealer can have sum from 1-21, at sum<12, should always hit
#also dealer follows policy that at sum<17, hit
policy_Dealer = np.zeros(22)
policy_Dealer[17] = 1
policy_Dealer[18] = 1
policy_Dealer[19] = 1
policy_Dealer[20] = 1
policy_Dealer[21] = 1


def play(policy_Player, initialState):
	#Initialize
	playerSum = 0
	dealerSum = 0
	episode = []
	dealerFaceUpCard = 0
	dealerFaceDownCard = 0
	playerUsabilityAce = False
	dealerUsabilityAce = False

	#Init cards 
	playerSum, dealerFaceUpCard, playerUsabilityAce = initialState
	dealerFaceDownCard = get_card()

	state = [playerSum, dealerFaceUpCard, int(playerUsabilityAce)]

	dealerSum = card_value(dealerFaceUpCard) + card_value(dealerFaceDownCard)

	dealerAceUsability = 1 in (dealerFaceUpCard, dealerFaceDownCard)
	#if dealerSum > 21, then since dealer has only 2 cards, she must have 2 aces
	#dealerSum if > 21 must be 22
	if dealerSum>21:
		#use 1 ace as 1 and other as 11
		dealerSum -= 10

	#Game starts
	#Turn = player
	flag = 1
	while True:
		if flag==0:
			action = initialAction
			flag = 1
		else:
			action = policy_Player()
		episode.append([(playerSum, dealerFaceUpCard, int(playerUsabilityAce)), action])

		#if action = stick
		if action == 1:
			break

		#if action = hit
		card = get_card()
		playerAceCount = int(playerUsabilityAce)
		if card==1:
			playerAceCount += 1
			#at first use all aces as 11, if going bust, use it as 1
		playerSum += card_value(card)
		#bust
		while playerSum>21 and playerAceCount:
			playerSum -= 10
			playerAceCount -= 1

		if playerSum>21:
			return state, -1, episode
		#update playerAceUsability on the bases on aceCount.
		#note that aceCount can be only 0 or 1 as if aceCount>1, then playersum>21
		playerUsabilityAce = (playerAceCount == 1)


	#Turn = dealer
	while True:
		action = policy_Dealer[dealerSum]

		if action==1:
			break

		card = get_card()
		dealerAceCount = int(dealerUsabilityAce)
		if card==1:
			dealerAceCount += 1
			#at first use all aces as 11, if going bust, use it as 1
		dealerSum += card_value(card)
		#bust
		while dealerSum>21 and dealerAceCount:
			dealerAceCount -= 1
			dealerSum -= 10
		if dealerSum>21:
			return state, 1, episode
		#update dealerAceUsability on the bases on aceCount.
		#note that aceCount can be only 0 or 1 as if aceCount>1, then dealersum>21
		dealerUsabilityAce = (dealerAceCount == 1)

	#both player and dealer have decided to stick, winner is closest to 21
	if playerSum>dealerSum:
		return state, 1, episode
	elif playerSum==dealerSum:
		return state, 0, episode
	else:
		return state, -1, episode

def behaviourPolicy_Player():
	#randomly choose an action
	#cannot use np.random.randint(0,1) -  did not work for ordinary sampling!
	if np.random.binomial(1, 0.5) == 1:
	       return 1
	return 0

def FirstVisitMCOffPolicy_EvaluationOfState(numEpisodes):
	#behaviour policy - fixed - randomly pick an action - function defined for the same
	#target policy - fixed - stick if sum>=20
	targetPolicy_Player = np.zeros(22)
	targetPolicy_Player[20] = 1
	targetPolicy_Player[21] = 1

	#state to be evaluated - sum=13, dealerfaceupcard=2, aceusability=True
	evaluateState = [13,2,True]

	#initialize rhos and returns from the behaviour policy for all episodes
	rhos = np.zeros(numEpisodes)
	behaviourPolicy_returns = np.zeros(numEpisodes)

	#To evaluate V(evaulateState)

	for iterations in range(0, numEpisodes):
		#evaluate starting from state evaluateState using behaviour policy
		state, returns, episode = play(behaviourPolicy_Player, evaluateState)

		# rho = 1
		numerator = 1.0
		denominator = 1.0

		for (playerSum, dealerFaceUpCard, playerUsabilityAce), action in episode:
			if action == targetPolicy_Player[playerSum]:
				denominator *= 0.5
			else:
				numerator = 0.0
				break
		rho = numerator/denominator
		rhos[iterations] = rho
		behaviourPolicy_returns[iterations] = returns

	#weightedReturns = targetPolicy_returns
	weightedReturns = rhos * behaviourPolicy_returns

	weightedReturns = np.add.accumulate(weightedReturns)
	rhos = np.add.accumulate(rhos)

	ordinarySampling = weightedReturns / np.arange(1, numEpisodes+1)
	weightedSampling = np.zeros(numEpisodes)

	for iteration in range(0,numEpisodes):
		if rhos[iteration] != 0:
			weightedSampling[iteration] = weightedReturns[iteration] / rhos[iteration]

	return ordinarySampling, weightedSampling


def fig5_3():
	True_StateVal = -0.27726
	runs = 100
	numEpisodes = 10000
	error_OrdinarySampling = np.zeros(numEpisodes)
	error_WeightedSampling = np.zeros(numEpisodes)
	
	for run in range(0,runs):
		ordinarySampling, weightedSampling = FirstVisitMCOffPolicy_EvaluationOfState(numEpisodes)
		error_OrdinarySampling += np.power(ordinarySampling - True_StateVal, 2)
		error_WeightedSampling += np.power(weightedSampling - True_StateVal, 2)

	error_OrdinarySampling /= runs
	error_WeightedSampling /= runs

	plt.plot(np.arange(1,numEpisodes+1), error_OrdinarySampling, label='ordinary')
	plt.plot(np.arange(1,numEpisodes+1), error_WeightedSampling, label='weighted')
	plt.xscale('log')
	plt.legend()
	plt.show()



def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id

fig5_3()