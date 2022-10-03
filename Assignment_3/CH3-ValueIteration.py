import matplotlib.pylab as plt
import numpy as np

states = 8  # number of states
A = ['r', 'l','d']  # actions
actions = 3

# In case that reward can be different to be in one state with
# different actions we candefine them seperately, in our example both
# left and rith actions lead to same reward in individual state [State, State, action]
Reward = [[0, 0, 0], [2, 2, 2], [1, 1, 1], [-1, -1, -1], [3, 3, 3], [-3, -3, -3], [-7, -7, -7], [5, 5, 5]]
TransitionProbaility = [  # [Right, Left]
    [[0.0, 0.0, 0.0], [0.3, 0.7, 0.0], [0.7, 0.3, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.3, 0.6, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.6, 0.1], [
        0.6, 0.3, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.6, 0.3, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.3, 0.6, 0.1], [0.6, 0.3, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.3, 0.6, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.7, 0.1], [0.7, 0.3, 0.1]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
]
# states = 8  # number of states
# A = ['r', 'l']  # actions
# actions = 2

# # In case that reward can be different to be in one state with
# # different actions we candefine them seperately, in our example both
# # left and rith actions lead to same reward in individual state [State, State, action]
# Reward = [[0, 0], [2, 2], [1, 1], [-1, -1], [3, 3], [-3, -3], [-7, -7], [5, 5]]
# TransitionProbaility = [  # [Right, Left]
#     [[0.0, 0.0], [0.3, 0.7], [0.7, 0.3], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.3, 0.7], [
#         0.7, 0.3], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.3, 0.7], [0.7, 0.3], [0.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.3, 0.7], [0.7, 0.3]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [
#         0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
# ]
Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Value estimation of each state

# -------------------------------------

gamma = 0.9
bellman_factor = 0
delta = 0.01
ValueList = []
ValueList.append(Value)
for iteration in range(0, 100):
    print(str(iteration) + ': ', Value[0], Value[1], Value[2], Value[3],  Value[4],  Value[5],
          Value[6],  Value[7], 'bellman_factor (' + str(bellman_factor) + ')', sep=",    ")
    NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]
    for i in range(states): 
        for a in range(actions):    #check each actions
            value_temp = 0
            for j in range(states): #check probaility to each states
                value_temp += TransitionProbaility[i][j][a] * Value[j]
            value_temp *= gamma
            value_temp += Reward[i][a]
            NewValue[i] = round(max(NewValue[i], value_temp), 2)
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue
    ValueList.append(Value)
    if(bellman_factor < delta):
        break

# Determine the policy (One time iteration)
NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]
policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
Terminal = ['', '', '', 'T', '', '', 'T', 'T']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbaility[i][j][a] * Value[j]
        value_temp *= gamma
        value_temp += Reward[i][a]
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy[i] = 'T'
array = np.array(ValueList)
transposed_array = array.T
fig, ax = plt.subplots()
for i in range(transposed_array.shape[0]):
    ax.plot([i for i in range(transposed_array.shape[1])], transposed_array[i], label='linear')
plt.title('ValueIteration',fontsize=12,color='r')
plt.show()
            
print(ValueList)
print("Policy Iteration algoirthm's final policy is:", policy)
