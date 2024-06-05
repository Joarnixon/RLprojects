import gym
import matplotlib.pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from gym.wrappers.monitoring import video_recorder
from dqn import Agent
from env_preprocess import process_observation
from loading import load_checkpoint
from train import train

device = 'cpu'
training = False

env = gym.make("MountainCar-v0", render_mode='rgb_array')

if training == True:
  model = Agent(env.action_space.n)
  model, history = train(env=env, model=model, episodes=20, eps_start=0.4, alpha=1e-4, gamma=0.99, batch_size=32, target_model_update=500)
else:
  model = load_checkpoint('checkpoint.pth')
  model = model.to(device)

video = video_recorder.VideoRecorder(env, "after_training.mp4")

state, info = env.reset()

for i in range(1500):
  env.render()
  video.capture_frame()
  state = process_observation(state).to(device)
  action = model.select_action(state, eps=0.05)
  state, reward, done, info, _ = env.step(action)

video.close()
env.close()
