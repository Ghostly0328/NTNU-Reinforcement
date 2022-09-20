#Upper Confidence Bounds(UCB) algorithm

from random import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Data visualization library based on matplotlib
import scipy.stats as stats
import math
import pandas as pd


class UpperConfidenceBounds():
    def __init__(self, counts, values):
        self.counts = counts
        self.p = values
        self.Qt = [1 for col in range(len(values))] #ini history_p = [0, 0, 0...]
        self.history = [[] for col in range(len(values))]  #ini history = [[0], [0], [0]...]
        self.result = []
        self.Nt = np.array([0.0 for col in range(len(values))])
        return
    
    def function(self, t):

        Ut = ((2 * math.log(10, t)) / self.Nt) ** -1
        Action = np.argmax(self.Qt + Ut)
        return Action

    def UpdateVariable(self, ArgAction):
        self.Nt[ArgAction] += 1

if __name__ == '__main__':

    N = 1000
    p_bandits = [0.5, 0.1, 0.8, 0.9]

    d = len(p_bandits)
    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_reward = [0] * d
    total_reward = 0
    historyProbability = []

    UCB = UpperConfidenceBounds(N, p_bandits)

    for n in range(N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_reward[i] / numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = 1 if np.random.random() < p_bandits[ad] else 0
        sums_of_reward[ad] += reward
        total_reward += reward

        historyProbability.append(total_reward / (n + 1))

    output = pd.Series(ads_selected).head(1500).value_counts(normalize=True)
    print(output)
    print(numbers_of_selections)
    print(historyProbability[-1])
    plt.figure()
    plt.title('UCB')
    plt.xlabel('Episode')
    plt.ylabel('Total Probability')
    plt.ylim(0,1)
    plt.plot(historyProbability, color='green', label='Thompson')
    plt.grid(axis='x', color='0.80')
    plt.show()