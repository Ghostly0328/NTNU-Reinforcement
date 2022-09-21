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
        self.numbers_of_selections = [0] * len(values)
        self.sums_of_reward = [0] * len(values)
        self.historyAvgProbility = []
        return

    def select(self):
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

        self.sums_of_reward[choosen_p] += ans
        self.numbers_of_selections[choosen_p] += 1

        self.historyAvgProbility.append(sum(self.DivideTwoArray(self.sums_of_reward, self.numbers_of_selections)))
        return

    def cal_result(self):       
        x = []
        for n in range(self.counts):
            x.append(sum(self.result[:n]) / (n + 1) )

        print("Numbers of Selections: ", str(self.numbers_of_selections))
        print("Sums of Reward: ", str(self.sums_of_reward))
        
        return self.historyAvgProbility
    
    def DivideTwoArray(self, arrayA, arrayB):
        
        Ans = [0.0 for col in range(len(arrayA))]
        for i in range(len(arrayA)):
            if arrayB[i] == 0:
                Ans[i] = 0
            else:
                Ans[i] = arrayA[i] / arrayB[i]
        return Ans
    
    def plt_result(self, array):

        plt.figure()
        plt.title('Aveage Reward Comparision')
        plt.xlabel('Episode')
        plt.ylabel('Total Probability')
        #plt.ylim(0,1)
        plt.plot(array, color='green', label='Thompson')
        plt.grid(axis='x', color='0.80')
        plt.show()

if __name__ == '__main__':
    np.random.seed(20)
    N = 1000
    epsilon = 0.95
    p_bandits = [0.5, 0.1, 0.8, 0.9]

    Number_of_Bandits = len(p_bandits)  #Number of Bandits
    epsilon_Function = EpsilonGreedy(epsilon, N, p_bandits)     #ini EpsilonGreedy

    for t in range(N):
        choosen_p = epsilon_Function.select()
        epsilon_Function.update(choosen_p)

    output = epsilon_Function.cal_result()
    
    epsilon_Function.plt_result(output)
    print("Last Probability: " + str(max(output)))
    