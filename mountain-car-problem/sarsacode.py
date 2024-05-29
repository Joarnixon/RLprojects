import pickle
import random
import numpy as np
import gymnasium as gym

random.seed(18)

from tile_coding import *


class SemiGradientSarsa(object):
    def __init__(self, env: gym.Env, alpha=0.01, eps=0.1, gamma=1, n_tilings = 7, num_eps=100) -> None:
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.num_eps = num_eps
        self.n_tilings = n_tilings
        self.env = env
        
        # define weight vector
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(n_tilings**4,))

        # hash for tile coding
        self.tile_coding = IHT(n_tilings**4)
    
    def q(self, feature_vector):
        return np.dot(self.w, feature_vector)

    def train(self):
        episodes = 0
        total_reward = 0
        state, info = self.env.reset()
        action = self.select_action(state)
        print("Started training")
        while episodes < self.num_eps:
            feature_vector = self.hash_feature_vector(state, action)

            next_state, reward, terminated, trunctated, _info = self.env.step(action)
            
            total_reward += reward

            if terminated:
                self.update_weight_terminal(reward, feature_vector)
                state, info = self.env.reset()
                action = self.select_action(state)
                if episodes % 50 == 0:
                    print("episode:", episodes, 'completed', 'reward:', total_reward)
                total_reward = 0
                episodes += 1
                continue

            next_action = self.select_action(next_state)
            self.update_weight(reward, self.hash_feature_vector(next_state, next_action), feature_vector)
            
            state = next_state
            action = next_action
        print('Saving')
        self.save_params()

    def update_weight(self, reward, feature_vector_next, feature_vector):
        self.w += self.alpha * (reward + self.gamma * self.q(feature_vector_next) - self.q(feature_vector)) * feature_vector
        return

    def update_weight_terminal(self, reward, feature_vector):
        self.w += self.alpha * (reward - self.q(feature_vector)) * feature_vector
        return
        

    def select_action(self, state, eps_greedy=True):
        n_actions = self.env.action_space.n
        actions = range(n_actions)
        action_values = []
        for action in actions: 
            feature_vector = np.array(self.hash_feature_vector(state, action))
            action_values += [self.q(feature_vector)]

        if eps_greedy:
            if random.random() > self.eps:
                return np.argmax(action_values)
            else:
                return random.randint(0, n_actions-1)
        else:
            return np.argmax(action_values)

    def save_params(self):
        pickle.dump(self.w, open('weights.pkl', 'wb'))
        pickle.dump(self.tile_coding, open('tilings.pkl', 'wb'))
    
    def load_params(self):
        self.w = pickle.load(open('weights.pkl', 'rb'))
        self.tile_coding = pickle.load(open('tilings.pkl', 'rb'))

    def one_hot_encode(self, indices):
        size = len(self.w)
        one_hot_vec = np.zeros(size)
        for i in indices:
            one_hot_vec[i] = 1
        return one_hot_vec

    def hash_feature_vector(self, state, action):
        feature_ind = np.array(tiles(self.tile_coding, self.n_tilings, state.tolist(), [action]))
        feature_vec = self.one_hot_encode(feature_ind)
        return feature_vec