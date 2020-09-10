import pandas as pd
import numpy as np
import gym
from gym import spaces
import pickle
import random

class UserModel():
    def __init__(self, data):
        self.data = data    #data should be pd.read_csv
        #self.userId = 758   #Change the start user Id here
        #self.rating_index = 0
        self.ind = 0

    def generate_new_user(self):
        tp = self.data[self.ind]
        self.ind += 1
        self.ind %= len(data)
        record = []
        for i in tp:
            movie = i[0]
            value = i[1]
            if value:
                record.append(movie)
        '''record = []
        #print("Begin to collect record")
        #print(self.rating_index,self.data.iloc[self.rating_index].userId)
        while True:
            if self.userId > 1409:
                self.userId = 758   #Change the start and end user Id here
                self.rating_index = 0
            while(self.rating_index < 99224 and self.data.iloc[self.rating_index].userId == self.userId):
                if self.data.iloc[self.rating_index].rating > 3:
                    record += [self.data.iloc[self.rating_index].movieId]    
                self.rating_index += 1

            self.userId += 1 #prepare for the next user
            if len(record) > 30:
                break'''

        #print("User record is ", record)
        tp = max(int(0.1 * len(record)), 1)
        train = set(random.sample(record, tp))
        test = set(record) - train
        return train, test

    def reset(self):
        #self.userId = 758
        #self.rating_index = 0
        self.ind = 0

class GridworldEnv(gym.Env):
    def __init__(self, data, gridworld, n, user = False, init_times = 10, N = 30):
        self.data = data
        self.N = N

        self.size = n
        self.gridworld = gridworld

        #The left top most point is (0,0)
        self.action_space = spaces.Discrete(4)
        self.actions = ['up', 'down', 'left', 'right']
        self.actions_pos_dict = {'up':[-1,0], 'down':[1,0], 'right':[0,1], 'left':[0,-1],}

        #define the initial user
        self.user = user
        self.user_record = set([])
        self.user_model = UserModel(self.data)
        self.train_x = 0
        self.train_y = 0
        self.sim_table = np.empty([self.size, self.size])
        self.cnt = 0
        self.init_times = init_times
        self.positions = []
        self.rec = set([])
        self.prn = 0

        #Initialize
        self.reset()



    def get_item(self, i, j):
        #Get recommendations from gridworld
        recommendations = set(self.gridworld[i][j][1])
        return recommendations

    def get_user(self, i, j):
        #Get recommendations from gridworld
        return set(self.gridworld[i][j][0])

    def initial_pos(self, n):
        # initial the starting position in the gridworld
        for i in range(self.size):
            for j in range(self.size):
                self.sim_table[i, j] = self.sim(self.get_item(i,j), self.user_record)
        tmp = np.argpartition(self.sim_table.reshape(-1), -n)[-n:]
        self.positions = [(int(i/self.size), i%self.size) for i in tmp]

    def new_pos(self):
        if self.cnt == 0:
            self.user_record, self.user_test = self.user_model.generate_new_user()
            self.initial_pos(self.init_times)
            self.position = self.positions[0]
        else:
            self.position = self.positions[self.cnt]
        self.cnt += 1
        self.cnt %= self.init_times
    
    def sim(self, current_recommendation, next_recommendation):
        # calculate similarity
        similarity = 0
        intersection = current_recommendation & next_recommendation
        union = current_recommendation | next_recommendation
        similarity = len(intersection)/len(union)
        return similarity

    def CTR(self, recommendations):
        # calculate click through rate of the whole recommendations
        # User click item in recommendations/len(recommendations)
        number = 0
        #print(self.interaction_times)
        number = len(self.user_test & recommendations)
        self.recall += number/len(self.user_test)
        self.precision += number/len(recommendations)
        self.prn += 1
        return self.precision / self.prn, self.recall / self.prn

    def CTR_clear(self):
        self.prn = 0
        self.recall = 0
        self.precision = 0

    def update_train(self):
        self.train_y += 1
        if self.train_y == self.size:
            self.train_y = 0
            self.train_x += 1
            if self.train_x == self.size:
                self.train_x = 0

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
        
        current_user = self.get_user(self.position[0],self.position[1])
        next_user = self.get_user(next_position[0], next_position[1])
        current_recommendation = self.get_item(self.position[0],self.position[1])
        next_recommendation = self.get_item(next_position[0], next_position[1])
        reward = self.sim(current_user, next_user)

        state = next_position
        self.position = next_position

        info = None
        
        if self.user:
            tmp = self.rec | self.recommendations | current_recommendation

        if not self.user or len(tmp) < self.N:
            if self.user:
                self.recommendations = tmp
            #For a new user
            self.recommendations |= current_recommendation
            done = len(next_recommendation - self.recommendations) == 0
        else:
            tp2 = self.rec | self.recommendations
            tp = len(tp2)
            self.recommendations |= set(random.sample(current_recommendation - tp2, self.N - tp))
            done = True
        
        #Add current recommendations to the set

        if done:
            reward = 0
            if self.user:
                self.rec = self.rec | self.recommendations
                if len(self.rec) >= self.N:
                    self.cnt = 0
                if self.cnt == 0:
                    info = self.CTR(self.rec)   #info is the CTR
                    self.rec = set([])
                self.new_pos()
                state = self.position
            else:
                info = (self.reward_sum, self.turns)
                self.turns = int(0)
                self.reward_sum = 0
                self.position = (self.train_x, self.train_y)
                state = self.position
                self.update_train()
            self.recommendations = set([])
        else:
            self.reward_sum += reward
            self.turns += 1
        #Return state, reward and done.
        #Here, self.state is the next state
        return state, reward, done, info
    
    def reset(self):
        self.recommendations = set([])
        self.position = ()
        if self.user:
            self.precision = 0
            self.recall = 0
            self.user_record = set([])
            self.user_test = set([])
            self.user_model.reset()
            self.new_pos()
        else:
            self.reward_sum = 0
            self.turns = 0
            self.position = (self.train_x, self.train_y)
            self.update_train()

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table = self.q_table.append(
                pd.Series(
                    [0.]*len(self.actions),
                    index=self.q_table.columns,
                    name='terminal',
                )
            )

    def choose_action(self, observation, greedy = False):
        self.check_state_exist(observation)
        if greedy:
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
        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)

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
                    [1.9]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == "__main__":
    f2 = open("test","rb")
    data = pickle.load(f2)
    f2.close()
    f = open("grid","rb")
    grid = pickle.load(f)
    f.close()
    width = 20
    print("Finish first step")
    env = GridworldEnv(data, grid, width, False)
    state = env.position
    action_list = ['up', 'down', 'right', 'left']
    QL = QLearningTable(actions = action_list)
    print("State is ", state)
    
    reward_sum = 4
    turns = 5

    for episode in range(200000):
        done = False
        while not done:
            #Choose an action
            action = QL.choose_action(str(state))
            #Get reward
            next_state, reward, done, info = env.step(action)
            
            #Learning
            if not done:
                QL.learn(str(state), action, reward, str(next_state), done)
            else:
                QL.learn(str(state), action, 0, 'terminal', done)

            #Update state
            state = next_state
            
            if info is not None:
                reward_sum = reward_sum * 0.99 + info[0] * 0.01
                turns = turns * 0.99 + info[1] * 0.01
            
        if(episode%800 == 799):
            print("Episode: ", episode)
            print("Reward: ", reward_sum)
            print("Rounds: ", turns)
            print(QL.q_table.sum().sum())

    f = open("ql","wb")
    pickle.dump(QL, f)
    f.close()

    f = open("ql","rb")
    QL = pickle.load(f)
    f.close()

    env.user = True
    env.reset()

    for episode in range(50000):
        done = False
        while not done:
            #Choose an action
            action = QL.choose_action(str(state), greedy = False)
            #Get reward
            next_state, reward, done, info = env.step(action)

            #Update state
            state = next_state
            
            if info is not None:
                precision, recall = info
            
        if(episode%5000 == 999):
            print("Episode: ", episode)
            print(env.prn, "Precision: ", precision, ", Recall: ", recall)
            env.CTR_clear()
    

    print("End")