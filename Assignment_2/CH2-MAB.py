# Thompson sampling algorithm for the the multi-armed bandits (RL-course)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Data visualization library based on matplotlib
import scipy.stats as stats
import time

# # set up matplotlib
# is_ipython = 'inline' in plt.get_backend()
# if is_ipython:
#     from IPython import display

np.random.seed(20) # Numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness.

# The probability of winning (exact value for each bandit), you can add more bandits here
Number_of_Bandits = 4
p_bandits = [0.5, 0.1, 0.8, 0.9] # Color: Blue, Orange, Green, Red


def bandit_run(index):
    if np.random.rand() >= p_bandits[index]: #random  probability to win or lose per machine
        return 0 # Lose
    else:
        return 1 # Win

def plot_steps(distribution, step, ax):
    plt.figure(1)
    plot_x = np.linspace(0.000, 1, 200) # create sequences of evenly spaced numbers structured as a NumPy array. # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    ax.set_title(f'Step {step:d}')
    for d in distribution:
        y = d.pdf(plot_x)
        ax.plot(plot_x, y) # draw edges/curve of the plot
        ax.fill_between(plot_x, y, 0, alpha=0.1) # fill under the curve of the plot
    ax.set_ylim(bottom = 0) # limit plot axis

def plot_rewards(rewards):
    plt.figure(2)
    plt.title('Aveage Reward Comparision')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    #plt.ylim(0,1)
    plt.plot(rewards, color='green', label='Thompson')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()
  
t0 = time.time()

N = 1000 # number of steps for Thompson Sampling
bandit_runing_count = [1] * Number_of_Bandits   # Array for Number of bandits try times, e.g. [0. 0, 0]
bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0. 0, 0]


figure, ax = plt.subplots(4, 3, figsize=(9, 7)) # set the number of the plots in row and column and their sizes
ax = ax.flat # Iterator to plot

average_reward = []
total_p = []

for step in range(1, N):
    # Beta distribution and alfa beta calculation
    bandit_distribution = []
    for run_count, win_count in zip(bandit_runing_count, bandit_win_count): # create a tuple() of count and win
        bandit_distribution.append (stats.beta(a = win_count + 1, b = run_count - win_count + 1)) # calculate the main equation (beta distribution)

    prob_theta_samples = []
    # Theta probability sampeling for each bandit
    for p in bandit_distribution:
        prob_theta_samples.append(p.rvs(1)) #rvs method provides random samples of distibution

    # Select best bandit based on theta sample a bandit
    select_bandit = np.argmax(prob_theta_samples)

    # Run bandit and get both win count and run count
    bandit_win_count[select_bandit] += bandit_run(select_bandit) 
    bandit_runing_count[select_bandit] += 1
     
    if step == 3 or step == 11 or (step % 100 == 1 and step <= 1000) :
        plot_steps(bandit_distribution, step - 1, next(ax))

    # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
    # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_runing_count[0] + ...
    average_reward_list = ([n / m for n, m in zip(bandit_win_count, bandit_runing_count)])
    
    # Get average of all bandits into only one reward value
    averaged_total_reward = 0
    for avged_arm_reward in (average_reward_list):
        averaged_total_reward += avged_arm_reward
    average_reward.append(averaged_total_reward)

    total_p.append(sum(bandit_win_count) / sum(bandit_runing_count))

t1 = time.time()
print("Cost Time: ", str(t1 - t0))

plt.tight_layout() # Adjust the padding between and around subplots.
plt.show()
plot_rewards(average_reward)
print("Last One of Probaility: ", average_reward[-1])
