import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns

#states : playerSum(12-21), dealerFaceUpCard(Ace-10), usabilityAce(1-usable.0-unusable)

#Action: hit=0, sticks=1
actionSet = [0,1]
gamma = 1

#policy: from dealerSum to action
#dealer can have sum from 1-21, at sum<12, should always hit
#also dealer follows policy that at sum<17, hit
policy_Dealer = np.zeros(22)
policy_Dealer[17] = 1
policy_Dealer[18] = 1
policy_Dealer[19] = 1
policy_Dealer[20] = 1
policy_Dealer[21] = 1


def play(policy):
	#Initialize
	playerSum = 0
	dealerSum = 0
	episode = []
	dealerFaceUpCard = 0
	dealerFaceDownCard = 0
	playerUsabilityAce = False
	dealerUsabilityAce = False

	#Init cards player
	#always hit if playerSum is less than 12
	while playerSum<12:
		#returns card from 1-10
		card = get_card()
		#use ace as 11 instead of 1 since playersum<12
		playerSum += card_value(card)
		#player now has atleast 1 ace (as only then would <12 become >21)
		#player must now have playerSum=22 
		if playerSum>21:
			#can use the ace as 1 to avoid bust
			playerSum -= 10
		else:
			playerUsabilityAce |= (1 == card)

	#Init cards dealer
	dealerFaceUpCard = get_card()
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

	while True:
		action = policy()
		episode.append([(playerSum, dealerFaceUpCard, int(playerUsabilityAce)), action, 0])

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

def MCOffPolicyControl(numEpisodes):

	#playerSum, dealerFceUpCard, AceUsabilityPlayer, Action(0-stick,1-hit)
	Estimate_StateActionValue = np.zeros((10,10,2,2))
	C = np.ones((10,10,2,2))

	#behaviour policy - fixed - randomly pick an action - function defined for the same
	#target policy - fixed - stick if sum>=20
	targetPolicy_Player = np.zeros((10,10,2))

	for iterations in range(0, numEpisodes):
		#generate episode
		state, returns, episode = play(behaviourPolicy_Player)
		G = returns
		W = 1
		for (playerSum, dealerFaceUpCard, playerUsabilityAce), action, reward in reversed(episode):
			G = gamma*G + reward
			playerSum -= 12
			dealerFaceUpCard -= 1
			C[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][action] += W 
			Estimate_StateActionValue[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][action] += (W/C[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][action]) * (G - Estimate_StateActionValue[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][action])

			#update policy
			QFunForEachAction = np.zeros(len(actionSet))
			maxQFun = -math.inf
			for i in range(0, len(actionSet)):
				QFunForEachAction[i] = Estimate_StateActionValue[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][i]
				maxQFun = max(maxQFun, QFunForEachAction[i])
			maxAction = np.argmax(QFunForEachAction)
			targetPolicy_Player[playerSum][dealerFaceUpCard][int(playerUsabilityAce)] = maxAction
			
			if action != targetPolicy_Player[playerSum][dealerFaceUpCard][int(playerUsabilityAce)]:
				break
			W = W*(1/0.5)

	return Estimate_StateActionValue, targetPolicy_Player

def fig5_2():
	Estimate_StateActionValue, targetPolicy_Player = MCOffPolicyControl(500000)
	Estimate_StateValue = np.zeros((10,10,2))
	pi = np.zeros((10,10,2))
	#v* = max a (q*)
	for i1 in range(0,10):
		for i2 in range(0,10):
			for i3 in range(0,2):
				maxtemp = -math.inf
				for i4 in range(0,2):
					maxtemp = max(maxtemp, Estimate_StateActionValue[i1,i2,i3,i4])
				Estimate_StateValue[i1,i2,i3] = maxtemp

	playerSum = np.arange(12,22)
	dealerFaceUpCard = np.arange(1,11)

	fig = plt.figure()
	axs = plt.axes(projection='3d')
	x,y = np.meshgrid(dealerFaceUpCard, playerSum)
	axs.plot_wireframe(x,y,Estimate_StateValue[:,:,1]) #0-unusable ace, 1-usable ace
	plt.show()




def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id

fig5_2()