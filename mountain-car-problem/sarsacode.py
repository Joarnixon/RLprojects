import pickle
import random
import numpy as np
import gymnasium as gym
from itertools import count

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
        self.w = np.zeros(n_tilings**4)

        # hash for tile coding
        self.tile_coding = IHT(n_tilings**4)
    
    def q(self, feature_vector):
        return np.dot(self.w, feature_vector)

    def train(self, n):
        episodes = 0
        best_reward = -10000000
        print("Started training")
        history = []
        while episodes < self.num_eps:
            actions_store = []
            states_store = []
            rewards_store = []

            total_reward = 0
            state, info = self.env.reset()
            action = self.select_action(state)

            actions_store += [action]
            states_store += [state]

            for t in count(0, 1):
                feature_vector = self.hash_feature_vector(state, action)

                next_state, reward, terminated, trunctated, _info = self.env.step(action)
                
                states_store += [next_state]
                rewards_store += [reward]

                total_reward += reward

                if terminated:
                    self.update_weight_terminal(reward, feature_vector)
                    history += [total_reward]
                    if episodes % 2 == 0:
                        print("episode:", episodes, 'completed', 'reward:', total_reward)
                    if total_reward > best_reward:
                        self.save_params()
                        best_reward = total_reward

                    episodes += 1
                    break

                next_action = self.select_action(next_state)
                actions_store += [next_action]

                state = next_state
                action = next_action

                tau = t - n + 1
                if tau >= 0:
                    self.update_weight(tau, n, states_store, actions_store, rewards_store)
        print('Best achieved and saved:', best_reward)
                
        return history
                

        print('Best reward achieved:', best_reward)
    
    def update_weight(self, tau, n, store_states, store_actions, store_rewards):
        # print(len(store_rewards), len(store_actions), len(store_states), tau+n+1, tau)
        G = sum(self.gamma**(i - tau - 1) * store_rewards[i-1] for i in range(tau+1, tau+n+1))
        G += self.gamma ** n * self.q(self.hash_feature_vector(store_states[tau + n], store_actions[tau + n]))

        self.w += self.alpha * (G - self.q(self.hash_feature_vector(store_states[tau], store_actions[tau]))) * self.hash_feature_vector(store_states[tau], store_actions[tau])

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