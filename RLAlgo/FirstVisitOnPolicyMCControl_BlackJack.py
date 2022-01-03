import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns

#states : playerSum(12-21), dealerFaceUpCard(Ace-10), usabilityAce(1-usable.0-unusable)

#Action: hit=0, sticks=1
actionSet = [0,1]
epsilon = 0.01

#policy: from dealerSum to action
#dealer can have sum from 1-21, at sum<12, should always hit
#also dealer follows policy that at sum<17, hit
policy_Dealer = np.zeros(22)
policy_Dealer[17] = 1
policy_Dealer[18] = 1
policy_Dealer[19] = 1
policy_Dealer[20] = 1
policy_Dealer[21] = 1


def play(policy_Player, initialState, initialAction):
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
	flag = 0
	while True:
		if flag==0:
			action = initialAction
			flag = 1
		else:
			action = policy_Player[playerSum-12][dealerFaceUpCard-1][int(playerUsabilityAce)]
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



def FirstVisitOnPolicyMCControl(numEpisodes):
	#randomly allocated a policy
	#policy: from playerSum to action
	#player can have sum from 1-21, at sum<12, should always hit
	#also player follows policy that at sum<20, hit
	policy_Player = np.zeros((10,10,2))

	#playerSum, dealerFceUpCard, AceUsabilityPlayer, Action(0-stick,1-hit)
	Estimate_StateActionValue = np.zeros((10,10,2,2))
	count_StateActionVisited = np.ones((10,10,2,2))
	for iteration in range(0,numEpisodes):
		#initial state,action to be picked randomly
		initialState = [np.random.choice(range(12,22)), np.random.choice(range(1,11)), bool(np.random.choice(range(0,2)))]
		initialAction = np.random.choice(actionSet)
		#update policy
		#policy_Player = updatePolicy()
		#generate episode starting from intialState,intialAction using updated policy
		startingState, returns, episode = play(policy_Player, initialState, initialAction)
		firstVisitCheck = set()
		
		for (playerSum, dealerFaceUpCard, playerUsabilityAce), action in episode:
			#1st visit would be playerSum-12
			playerSum -= 12
			dealerFaceUpCard -= 1

			if (playerSum, dealerFaceUpCard, playerUsabilityAce, action) in firstVisitCheck:
				continue
			else:
				firstVisitCheck.add((playerSum, dealerFaceUpCard, playerUsabilityAce, action))

			Estimate_StateActionValue[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][int(action)] += returns
			count_StateActionVisited[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][int(action)] += 1
			#update policy
			QFunForEachAction = np.zeros(len(actionSet))
			maxQFun = -math.inf
			for i in range(0, len(actionSet)):
				QFunForEachAction[i] = Estimate_StateActionValue[playerSum][dealerFaceUpCard][int(playerUsabilityAce)][i]
				maxQFun = max(maxQFun, QFunForEachAction[i])
			maxAction = np.argmax(QFunForEachAction)
			#choose epsilon greedily 
			# wp epsilon select random action, wp 1-epsilon select greedy action
			if np.random.uniform(0,1)<epsilon:
				action = np.random.randint(0,len(actionSet),dtype='int')
			else:
				action = maxAction
			policy_Player[playerSum][dealerFaceUpCard][int(playerUsabilityAce)] = action
			
	return Estimate_StateActionValue/count_StateActionVisited

def fig5_2():
	Estimate_StateActionValue = FirstVisitOnPolicyMCControl(800000)
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

	#pi* = max a (q*)
	for i1 in range(0,10):
		for i2 in range(0,10):
			for i3 in range(0,2):
				pi[i1,i2,i3] = np.argmax(Estimate_StateActionValue[i1,i2,i3,:])
	playerSum = np.arange(12,22)
	dealerFaceUpCard = np.arange(1,11)

	#fig = plt.figure()
	#axs = plt.axes(projection='3d')
	#meshgrid of dealerFaceUpCard and playerSum
	#x,y = np.meshgrid(dealerFaceUpCard, playerSum)
	#axs.plot_wireframe(x,y,Estimate_StateValue[:,:,0]) #0-unusable ace, 1-usable ace
	#plt.show()

	_, axes = plt.subplots(1,1)
	plt.subplots_adjust(wspace=0.1, hspace=0.2)
	fig = sns.heatmap(np.flipud(pi[:,:,0]),cmap="YlGnBu", ax=axes, xticklabels=range(1, 11),yticklabels=list(reversed(range(12, 22))))
	
	plt.show()



def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id

fig5_2()