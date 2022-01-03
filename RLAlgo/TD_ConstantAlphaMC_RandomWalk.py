import numpy as np 
import matplotlib.pyplot as plt 

def TD(stateValue, alpha, batch=False):
    episode = [3]
    reward = [0]
    currentState = 3
    
    while True:
        oldState = currentState
        if Prob_LeftAction==np.random.binomial(1, 0.5):
            currentState -=1
        else:
            currentState +=1
        episode.append(currentState)
        #TD(0) update rule - online update
        if batch==False:
            stateValue[oldState] += alpha*(stateValue[currentState] - stateValue[oldState])
        #if terminating state reached
        if currentState==0 or currentState==6:
            break
        reward.append(0)
        
    return episode, reward

#every-visit MC
def MC(stateValue, alpha, batch=False):
    episode = [3]
    returns = 0.0
    currentState = 3
    
    while True:
        if Prob_LeftAction==np.random.binomial(1, 0.5):
            currentState -=1
        else:
            currentState +=1
        episode.append(currentState)
        #if terminating state reached
        if currentState==0:
            break
        elif currentState==6:
            returns=1.0
            break
     
    if batch==False:
        #MC update rule - offline update
        for state1 in episode[:-1]:
            stateValue[state1] += alpha*(returns - stateValue[state1]) 
    
    length_episode = len(episode)-1
    return episode, [returns]*length_episode

def batchProcessing_TD(numEpisodes, alpha):
    runs = 100
    RMSError = np.zeros(numEpisodes)

    for run in range(0, runs):
        stateValues = np.copy(Estimate_StateValue)
        stateValues[1:6] = -1
        errors = []
        episodes = []
        rewards = []

        for i in range(0, numEpisodes):
            episode, reward = TD(stateValues, alpha, True)
            episodes.append(episode)
            rewards.append(reward)

            while True:
                #feed the episodes seen so far
                updates = np.zeros(7)
                for episode, reward in zip(episodes, rewards):
                    for j in range(0, len(episode)-1):
                        updates[episode[j]] += alpha*(reward[j] + gamma*(stateValues[episode[j+1]]-stateValues[episode[j]]))

                if np.sum(np.abs(updates)) < 1e-3:
                    break
                #batch update
                stateValues += updates
            #rms error
            errors.append(np.sqrt(np.sum(np.power(True_StateValue - stateValues, 2)) / 5.0))
        RMSError += np.asarray(errors)
    RMSError /= runs
    return RMSError

def batchProcessing_MC(numEpisodes, alpha):
    runs = 100
    RMSError = np.zeros(numEpisodes)

    for run in range(0, runs):
        stateValues = np.copy(Estimate_StateValue)
        stateValues[1:6] = -1
        errors = []
        episodes = []
        rewards = []

        for i in range(0, numEpisodes):
            episode, reward = MC(stateValues, alpha, True)
            episodes.append(episode)
            rewards.append(reward)

            while True:
                #feed the episodes seen so far
                updates = np.zeros(7)
                for episode, reward in zip(episodes, rewards):
                    for j in range(0, len(episode)-1):
                        updates[episode[j]] += alpha*(reward[j] - stateValues[episode[j]])

                if np.sum(np.abs(updates)) < 1e-3:
                    break
                #batch update
                stateValues += updates
            #rms error
            errors.append(np.sqrt(np.sum(np.power(True_StateValue - stateValues, 2)) / 5.0))
        RMSError += np.asarray(errors)
    RMSError /= runs
    return RMSError


def figure1new():
    stateValues = np.copy(Estimate_StateValue)
    plt.figure(1)
    for iterations in range(0,101):
        if iterations==0 or iterations==1 or iterations==10 or iterations==100:
            temp = [stateValues[i] for i in range(1,6)]
            plt.plot([1,2,3,4,5],temp, label=str(iterations))
        TD(stateValues, 0.1)
    temp = [True_StateValue[i] for i in range(1,6)]
    plt.plot([1,2,3,4,5],temp, label='True Values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()

def figure2():
    alpha_MC = [0.01, 0.02, 0.03, 0.04]
    alpha_TD = [0.05, 0.1, 0.15]
    alphaList = [0.05, 0.1, 0.15, 0.01, 0.02, 0.03, 0.04]
    
    for iteration in range(0,7):
        rmsError = np.zeros(101)
        method = ''
        ls = ''
        if iteration<3:
            method = 'TD'
            ls = 'solid'
        else:
            method = 'MC'
            ls = 'dashdot'
        
        for run in range(100):
            eL = []
            currentStateValue = np.copy(Estimate_StateValue)
            for i in range(0, 101):
                eL.append(np.sqrt(np.sum(np.power(True_StateValue - currentStateValue, 2))/5.0))
                if method == 'TD':
                    TD(currentStateValue, alphaList[iteration])
                else:
                    MC(currentStateValue, alphaList[iteration])
            rmsError += np.asarray(eL)
        rmsError /= 100
        st = str(alphaList[iteration])+" "
        plt.plot(rmsError, linestyle=ls, label='alpha= '+st+method)
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.legend()
    plt.show()

def figure3():
    TDErrors = batchProcessing_TD(101, 0.001)
    MCErrors = batchProcessing_MC(101, 0.001)

    plt.plot(np.arange(100+1), TDErrors, label='td')
    plt.plot(np.arange(100+1), MCErrors, label='mc')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()
    plt.show()


# Initialization
True_StateValue = np.zeros(7)
True_StateValue[6] = 1
for i in range(1,6):
    True_StateValue[i] = i/6.0

Estimate_StateValue = np.zeros(7)
Estimate_StateValue[6] = 1
for i in range(1,6):
    Estimate_StateValue[i] = 1/2.0
    
# initial policy
Prob_RightAction = 1
Prob_LeftAction = 0

gamma = 1

figure3()