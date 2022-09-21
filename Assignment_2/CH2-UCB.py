#Upper Confidence Bounds(UCB) algorithm

from array import array
from random import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


class UpperConfidenceBounds():
    def __init__(self, counts, values):     #setting
        self.N = counts
        self.p_bandits = values
        self.d = len(values)
        self.bandits_selected = []
        self.numbers_of_selections = [0] * self.d
        self.sums_of_reward = [0] * self.d
        self.total_reward = 0

    def start(self):

        historyProbability = []
        historyAvgProbility = []

        for n in range(self.N):
            bandits = 0 #choose number of bandits
            max_upper_bound = 0
            for i in range(0, self.d):
                if (self.numbers_of_selections[i] > 0):
                    average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                    delta_i = math.sqrt(2 * math.log(n+1) / self.numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    bandits = i
            self.bandits_selected.append(bandits)
            self.numbers_of_selections[bandits] += 1
            reward = self.BanditsWinOrLose(bandits)
            self.sums_of_reward[bandits] += reward
            self.total_reward += reward

            historyProbability.append(self.total_reward / (n + 1))
            historyAvgProbility.append(sum(self.DivideTwoArray(self.sums_of_reward, self.numbers_of_selections)))

        output = pd.Series(self.bandits_selected).head(1500).value_counts(normalize=True)

        print(output)
        print("Number of Selections: ", str(self.numbers_of_selections))
        print("Number of reward: ", str(self.sums_of_reward))
        print("Last One of Probaility: ", str(historyProbability[-1]))
        print(historyAvgProbility)

        self.Plt(historyAvgProbility)

    def BanditsWinOrLose(self, idx):
        return 1 if np.random.random() < self.p_bandits[idx] else 0

    def Plt(self, ShowArray):

        plt.figure()
        plt.title('UCB')
        plt.xlabel('Episode')
        plt.ylabel('Probability: 1~n')
        #plt.ylim(0,1)
        plt.plot(ShowArray, color='green', label='Thompson')
        plt.grid(axis='x', color='0.80')
        plt.show()

    def DivideTwoArray(self, arrayA, arrayB):
        
        Ans = [0.0 for col in range(len(arrayA))]
        for i in range(len(arrayA)):
            if arrayB[i] == 0:
                Ans[i] = 0
            else:
                Ans[i] = arrayA[i] / arrayB[i]
        return Ans

if __name__ == '__main__':

    np.random.seed(20)  #control stocastic random
    N = 1000
    p_bandits = [0.5, 0.1, 0.8, 0.9]
    
    UCB = UpperConfidenceBounds(N, p_bandits)
    UCB.start()