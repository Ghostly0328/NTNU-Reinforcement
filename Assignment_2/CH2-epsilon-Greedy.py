#Epsilon Greedy Algorithm site: https://blog.csdn.net/lafengxiaoyu/article/details/102634543

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Data visualization library based on matplotlib
import scipy.stats as stats

class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.p = values
        self.history_p = [0.0 for col in range(len(values))] #ini history_p = [0, 0, 0...]
        self.history = [[] for col in range(len(values))]  #ini history = [[0], [0], [0]...]
        self.result = []
        return

    def select(self):
        #print(np.random.random())
        if np.random.random() < self.epsilon:
            return np.argmax(self.history_p)
        else:
            return np.random.randint(len(self.history_p))
    
    def bandit_run(self, chosen_p):
        if np.random.rand() >= self.p[chosen_p]:
            return 0    # Lose
        else:
            return 1    # Win

    def update(self, chosen_p):
        ans = self.bandit_run(chosen_p)
        self.history[chosen_p].append(ans)

        for idx, array in enumerate(self.history_p):
            if len(self.history[idx]) == 0:
                self.history_p[idx] = 0
            else:
                self.history_p[idx] = sum(self.history[idx]) / len(self.history[idx])

        self.result.append(ans)

        #print(self.history_p)
        return

    def cal_result(self):
        
        x = []

        for n in range(self.counts):
            x.append(sum(self.result[:n]) / (n + 1) )

        plt.figure()
        plt.title('Aveage Reward Comparision')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.ylim(0,1)
        plt.plot(x, color='green', label='Thompson')
        plt.grid(axis='x', color='0.80')
        plt.legend(title='Parameter where:')
        plt.show()

if __name__ == '__main__':

    np.random.seed(20) # Numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness.

    N = 1000
    p_bandits = [0.5, 0.1, 0.8, 0.9] # Color: Blue, Orange, Green, Red

    Number_of_Bandits = len(p_bandits)

    epsilon = EpsilonGreedy(0.95, N, p_bandits)

    for t in range(N):
        chosen_p = epsilon.select()
        epsilon.update(chosen_p)    #update history
    epsilon.cal_result()
    