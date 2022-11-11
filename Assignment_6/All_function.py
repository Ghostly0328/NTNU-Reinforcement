from Expected_SARSA import Expected_SARSA
from Qlearning import Qlearning
from SARSA import SARSA

import gym
import matplotlib.pyplot as plt

def plot(rewards1, rewards2, rewards3):
    plt.figure(2)
    plt.title('All of Aveage Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.plot(rewards1, color='green', label='Expected_SARSA')
    plt.plot(rewards2, color='red', label='Qlearning')
    plt.plot(rewards3, color='yellow', label='SARSA')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()
    
if __name__ == "__main__":
        
    alpha = 0.1 # learning rate
    gamma = 0.8 # discount factor
    epsilon = 0.01 # epsilon greedy exploration-explotation (smaller more random)
    episodes = 200
    
    EPS_START = 0.001
    EPS_END = 1
    EPS_DECAY = 10 

    max_steps = 2500 # to make it infinite make sure reach objective

    Expected_SARSA_timestep_reward = Expected_SARSA(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests = 1)

    Qlearning_timestep_reward2 = Qlearning(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests = 0)
    
    SARSA_timestep_reward3 = SARSA(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests = 0)
    
    plot(Expected_SARSA_timestep_reward, Qlearning_timestep_reward2, SARSA_timestep_reward3)