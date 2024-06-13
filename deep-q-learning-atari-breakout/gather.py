import torch
import cv2
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from gym.core import ObservationWrapper
from gym.wrappers import FrameStack
from gym.utils import play

class Preprocessing(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.image_size = (64, 64)
        self._gray_scale_rule = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32).T.unsqueeze(1)

    def _gray_scale(self, img):
        return torch.matmul(img, self._gray_scale_rule.reshape(-1, 1))

    def observation(self, img):
        img = img[90:200, 3:157]
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32)
        img = self._gray_scale(img).squeeze(-1)
        img = img.unsqueeze(0) / 255
        return img

def AtariWrap(env):
    env = Preprocessing(env)
    env = FrameStack(env, 4)
    return env

class Dataset:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.data_states = torch.empty((1, 4, 1, 64, 64), dtype=torch.float32)
        self.data_actions = torch.empty((1), dtype=torch.float32)
        self.frames = torch.empty((1, 1, 64, 64), dtype=torch.float32)
        self.n = 0

    def _append_data(self, frames, action):
        self.data_states = torch.cat((self.data_states, frames))
        self.data_actions = torch.cat((self.data_actions, action))
        self.n += 1
    
    def _process_observation(self, obs):
        return torch.tensor(np.array([frame.to('cpu') for frame in obs._frames]), dtype=torch.float32).unsqueeze(0)

    def append_frames(self, frames, action):
        frames = self._process_observation(frames)
        action = torch.tensor([action], dtype=torch.int64)

        self._append_data(frames, action)
    
    def save(self):
        np.savez('data_states.npz', self.data_states.numpy())
        np.savez('data_actions.npz', self.data_actions.numpy())


collectable = Dataset(4)

def callback(obs_t, obs_tp1, action, rew: float, terminated: bool, truncated: bool, info: dict):
    if action != 0:
        collectable.append_frames(obs_tp1, action)
        print(collectable.n)
    return

env = gym.make("BreakoutNoFrameskip-v4", render_mode='rgb_array')
env = AtariWrap(env)
play.play(env, callback=callback)
collectable.save()

# example = np.load('data_states.npz')['arr_0']
# print(example.shape)