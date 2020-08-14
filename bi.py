import pandas as pd
import numpy as np
import gym
from gym import spaces
import pickle

class UserModel():
    def __init__(self, data):
        self.data = data    #data should be pd.read_csv
        self.userId = 758   #Change the start user Id here
        self.rating_index = 0

    def generate_new_user(self):
        record = []
        #print("Begin to collect record")
        while(self.data.iloc[self.rating_index].userId == self.userId):
            if self.data.iloc[self.rating_index].rating > 3:
                record += [self.data.iloc[self.rating_index].movieId]    
            self.rating_index += 1
        self.userId += 1 #prepare for the next user
        if self.userId > 1409: self.userId = 758   #Change the start and end user Id here

        #print("User record is ", record)
        return record

    def reset(self):
        self.userId = 758
        self.rating_index = 0

class GridworldEnv(gym.Env):
    def __init__(self, data, gridworld, n):
        self.data = data

        self.size = n
        self.gridworld = gridworld
        self.recommendations = set([])
        self.click_rate = 0
        self.interaction_times = 0


        #The left top most point is (0,0)
        self.action_space = spaces.Discrete(4)
        self.actions = ['up', 'down', 'left', 'right']
        self.actions_pos_dict = {'up':[-1,0], 'down':[1,0], 'right':[0,1], 'left':[0,-1],}

        #define the initial user
        self.position = ()
        self.user_record = []
        self.user_model = UserModel(self.data)

        #Initialize
        self.reset()



    def get_recommendation(self, i, j):
        #Get recommendations from gridworld
        recommendations = self.gridworld[i][j][1]
        return recommendations

    def initial_pos(self):
        # initial the starting position in the gridworld
        similarity = -10e9
        position = ()
        for i in range(self.size):
            for j in range(self.size):
                temp = self.Similarity(self.get_recommendation(i,j),self.user_record)
                if temp > similarity:
                    similarity = temp
                    position = (i,j)
        return position

    def new_user(self):
        self.user_record = self.user_model.generate_new_user()
        self.position = self.initial_pos()
    
    def Similarity(self, current_recommendation, next_recommendation):
        # calculate similarity
        similarity = 0
        intersection = set(current_recommendation) & set(next_recommendation)
        union = set(current_recommendation) | set(next_recommendation)
        similarity = len(intersection)/len(union)
        return similarity

    def CTR(self, recommendations):
        # calculate click through rate of the whole recommendations
        # User click item in recommendations/len(recommendations)
        self.interaction_times = self.interaction_times + 1
        number = 0
        for item in recommendations:
            if item in self.user_record:
                number = number + 1
        self.click_rate = self.click_rate + number/len(recommendations)
        return self.click_rate/self.interaction_times


    def step(self, action):
        # define the step
        #Limit next position
        pos0 = self.position[0]+self.actions_pos_dict[action][0]
        if(pos0 < 0): pos0 = 0
        if(pos0 > self.size - 1): pos0 = self.size - 1
        pos1 = self.position[1]+self.actions_pos_dict[action][1]
        if(pos1 < 0): pos1 = 0
        if(pos1 > self.size - 1): pos1 = self.size - 1
        next_position = (pos0, pos1)
        
        current_recommendation = self.get_recommendation(self.position[0],self.position[1])
        next_recommendation = self.get_recommendation(next_position[0], next_position[1])
        reward = self.Similarity(current_recommendation, next_recommendation)

        state = next_position
        self.position = next_position

        #Add current recommendations to the set
        self.recommendations = self.recommendations | set(current_recommendation)
        info = self.CTR(current_recommendation)   #info is the CTR

        #For a new user
        done = True
        for i in next_recommendation:
            if i not in self.recommendations:
                done = False
        if done is True:
            self.new_user()
            state = self.position
        #Return state, reward and done.
        #Here, self.state is the next state
        return state, reward, done, info
    
    def reset(self):
        self.recommendations = set([])
        self.click_rate = 0
        self.interaction_times = 0
        self.position = ()
        self.user_record = []
        self.user_model.reset()
        self.new_user()

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            #print("here, observation is ", observation)
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if done != True:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == "__main__":
    data = pd.read_csv('100kRatings2.csv')
    f = open("grid","rb")
    grid = pickle.load(f)
    width = 20
    print("Finish first step")
    env = GridworldEnv(data, grid, width)
    state = env.position
    action_list = ['up', 'down', 'right', 'left']
    QL = QLearningTable(actions = action_list)

    print("State is ", state)
    
    for episode in range(100000):
        done = False
        while not done:
            #Choose an action
            action = QL.choose_action(str(state))

            #Get reward
            next_state, reward, done, info = env.step(action)

            #Learning
            QL.learn(str(state), action, reward, str(next_state), done)

            #Update state
            state = next_state
        if(episode%1000 == 0):
            print("CTR is ", info)
    print("End")

