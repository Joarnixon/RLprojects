import torch
import torch.nn as nn
import random

class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n = n_actions

        self.linear1 = nn.Linear(2, 100, dtype=torch.float32)
        self.relu1 = nn.ReLU()
        self.linear3 = nn.Linear(100, n_actions, dtype=torch.float32)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear3(x)        
        return x

    def select_action(self, observation, eps_greedy=True, eps=0.2):
        if eps_greedy:
            if random.random() > eps:
                q = self(observation)
                max_q = torch.argmax(q)
                return max_q.item()
            else:
                return random.randint(0, self.n-1)
        else:
            q = self(observation)
            max_q = torch.argmax(q)
            return max_q.item()

    def get_q_value(self, state, action):
        if len(action.shape) == 0:
            return self(state)[action.item()]
        else:
            action = action.tolist()
            return self(state)[range(0, len(action)), action]