import time
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

random.seed(10)

from sarsacode import SemiGradientSarsa

env_ = gym.make("MountainCar-v0", render_mode=None)

# n_mean = 10
# hists1 = []
# hists2 = []
# env_mean = gym.make("MountainCar-v0")

# for i in range(n_mean):
#     state, info = env_mean.reset()
#     sarsa = SemiGradientSarsa(env_mean, alpha=0.01/8, eps=0.1, gamma=0.95, n_tilings = 8, num_eps=20)
#     hists1 += [sarsa.train(n=2)]

#     state, info = env_mean.reset()
#     sarsa = SemiGradientSarsa(env_mean, alpha=0.01/8, eps=0.1, gamma=0.95, n_tilings = 8, num_eps=20)
#     hists2 += [sarsa.train(n=4)]

# hists1 = -np.mean(hists1, axis=0)
# hists2 = -np.mean(hists2, axis=0)
# plt.plot(hists1, color='r')
# plt.plot(hists
# 2, color='b')
# plt.show()
sarsa = SemiGradientSarsa(env_, alpha=0.01/8, eps=0.1, gamma=0.95, n_tilings = 8, num_eps=100)
sarsa.train(n=10)
sarsa.load_params()

env = gym.make("MountainCar-v0", render_mode='human')
state, info = env.reset()

for i in range(200):
    action = sarsa.select_action(state, eps_greedy=False)
    next_state, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.001)

    state = next_state

env.close()