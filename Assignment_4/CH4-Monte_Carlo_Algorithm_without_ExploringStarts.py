# Problem: A robot needs to go from Start to Target
# Reward: +100 for target, -1 for each other step
# Q-value: for simplicity zero
# Initial Q values are all zero

from operator import index
from matplotlib.pyplot import grid
import numpy as np
import random
#隨機種子
np.random.seed(9527)
random.seed(9527)

class Environment():
    def __init__(self):
        self.rows = 4   #列
        self.cols = 6   #行
        self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
                           [ "s6",  "s7",  "s8",  "s9",   "W", "s10"],
                           ["s11",   "W", "s12",   "W", "s13", "s14"],
                           ["s15", "s16", "s17", "s18", "s19", "s20"]] #T: Target, W: Wall
        self.action_to_number = {"up": 0, "right":1, "down":2, "left":3}    #行為
        self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1]} #行為座標轉換
        self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left"}
        self.invalid_start = ["T", "W"]

    def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
        current_state_coordinates = state_coordinates
        next_state_coordinates = state_coordinates + self.action_dict[action]

        if next_state_coordinates[0] < 0 or next_state_coordinates[0] > 3 or next_state_coordinates[1] < 0 or next_state_coordinates[1] > 5:# Out of board
            return current_state_coordinates
        next_state = self.grid_world[next_state_coordinates[0]][next_state_coordinates[1]]
        if next_state == "W": #Hit the wall 
            return current_state_coordinates
        return next_state_coordinates

class Monte_Carlo():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = 10000
        self.gamma = 0.9
        self.Horizon = 10 #Max episode_length
        self.epsilon = 0.5

        self.Q_values = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]

    def generate_initial_state(self): #Generate random state
        while True:
            state_row = np.random.randint(self.env.rows)
            state_col = np.random.randint(self.env.cols)
            if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
                continue
            else:
                break
        return np.array([state_row, state_col])

    def generate_random_action(self):
        action = self.env.direction_dict[np.random.randint(4)]
        while (self.env.transfer_state(self.current_state_coordinates, action) == self.current_state_coordinates).all():
                action = self.env.direction_dict[np.random.randint(4)]
        return action

    def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
        Q_value = []
        valid_actions = []
        state_coordinate = np.array(state_coordinate)
        indexes = []
        for action in self.env.action_dict.keys():
            if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
                valid_actions.append(action)

        for valid_action in valid_actions:
            Q_value.append(self.Q_values[(tuple(state_coordinate), valid_action)])
        max_value = max(Q_value)

        for valid_action in valid_actions:
            if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                indexes.append(self.env.action_to_number[valid_action])
        # indexes = [index for index,x in enumerate(Q_value) if x == max_value]
        return self.env.direction_dict[random.choice(indexes)]

    def print_Qvalue(self, state_coordinate): #For debugging
        Q_value = []
        for action in self.env.action_dict.keys():
            Q_value.append(self.Q_values[(tuple(state_coordinate), action)])
        return Q_value

    def iter(self): #Main loop
        for iterration in range(self.Max_iteration):
            episode = []
            self.current_state_coordinates = self.generate_initial_state()
            action = self.generate_random_action()
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    episode.append([self.current_state_coordinates, action, reward])
                    break
                episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                self.current_state_coordinates = next_state_coordinates
                action = self.policy(self.current_state_coordinates)
            G = 0
            
            for h in range(len(episode)-1, -1, -1): #Iterate through H-1, H-2,...,0
                coordinate, action, reward = episode[h]
                G = self.gamma*G + reward
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + G)/(visited_count + 1)
                visited_count += 1
                self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]
                
                max_value = max(self.Q_values[(tuple(episode[h][0]), episode[h][1])])
                for n_Q_value in range(len(self.Q_values[(tuple(episode[h][0]), episode[h][1])])):
                    #Pretect division by zero 
                    self.Q_values[(tuple(episode[h][0]), episode[h][1])] = 1e-5 if self.Q_values[(tuple(episode[h][0]), episode[h][1])] == 0 else self.Q_values[(tuple(episode[h][0]), episode[h][1])]

                    if self.Q_values[(tuple(episode[h][0]), episode[h][1])] == max_value:
                        self.Q_values[(tuple(episode[h][0]), episode[h][1])] = 1 - self.epsilon + self.epsilon/abs(self.Q_values[(tuple(episode[h][0]), episode[h][1])])
                    else:
                        self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.epsilon/abs(self.Q_values[(tuple(episode[h][0]), episode[h][1])])  

    def render(self): #Show results
        output = self.env.grid_world
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    action = self.policy((row,col))
                    output[row][col] = action

        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")

    def demo(self): #Slides example
        self.Horizon = 6
        initial_state_actions = [[[1,1], "left"], [[1,2], "right"], [[2,2], "up"], [[1,1], "up"]]
        for state_action in initial_state_actions:
            episode = []
            self.current_state_coordinates = np.array(state_action[0])
            action = state_action[1]
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    episode.append([self.current_state_coordinates, action, reward])
                    break
                episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                self.current_state_coordinates = next_state_coordinates
                action = self.policy(self.current_state_coordinates)
            G = 0
            print("episode sequence:", episode)
            for h in range(len(episode)-1, -1, -1): #Iterate H-1, H-1,...,0
                coordinate, action, reward = episode[h]
                G = self.gamma*G + reward
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + G)/(visited_count + 1)
                visited_count += 1
                self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]

            print("Returns:")
            for h in range(len(episode)-1, -1, -1):
                coordinate, action, reward = episode[h]
                print("Coordinate:", coordinate,"\tAction:", action, "\tReturn:", self.returns_dict[(tuple(coordinate), action)])
            print("")

if __name__ == "__main__":
    print("Find optimal policy using Monte Carlo algorithm with exploring starts")
    monte = Monte_Carlo()
    print("Before (random policy), T = Target, W = Wall")
    monte.render()
    monte.iter()
    print("\nOptimal policy, T = Target, W = Wall")
    monte.render()
    print("\n\n\n")

    print("************************Slides example***********************")
    monte = Monte_Carlo()
    initial_policy = [((0,1),"left"), ((0,2), "left"), ((0,3), "left"), ((0,4), "right"), ((0,5), "left"),
                      ((1,0), "right"), ((1,1), "right"), ((1,2), "down"), ((1,3), "up"), ((1,5), "up"),
                      ((2,0), "up"), ((2,2), "down"), ((2,4), "right"), ((2,5), "up"),
                      ((3,0), "up"), ((3,1), "left"), ((3,2),"left"), ((3,3), "right"), ((3,4), "left"), ((3,5), "left")]
    for state_action in initial_policy:
        monte.Q_values[state_action] = 1
    print("Before (random policy), T = Target, W = Wall")
    monte.render()
    print("\n\nRunning the episodes...\n\n")
    monte.demo()
    print("After running the episodes, T = Target, W = Wall")
    monte.render()
    
    # print("Q_value",monte.print_Qvalue((3,1)))

        
    








