import torch
import torch.nn as nn
from torch.optim import Adam
import random
import numpy as np
from itertools import count
from env_preprocess import process_observation
import copy
from loading import save_checkpoint

loss_fn = nn.MSELoss()

def compute_q_loss(model, model_target, states, actions, rewards, next_states, gamma, alpha, device):
    with torch.no_grad():
        Q_target = torch.max(model_target(next_states.to(device)))
        
    first = model.get_q_value(states.to(device), actions)
    second = rewards.to(device) + gamma * Q_target
    return alpha * loss_fn(second, first)

def compute_q_loss_terminal(model, states, actions, rewards, alpha, device):
    first = model.get_q_value(states.to(device), actions)
    second = rewards.to(device)
    return alpha * loss_fn(second, first)

def train(env, model, episodes, eps_start=1, alpha=1e-3, gamma=0.99, buffer_size=100000, batch_size=32, target_model_update=1000, device='cpu'):
    n_done = 0
    print("Started training")
    history = []
    
    model_target = copy.deepcopy(model).to(device)
    model_target.dtype = torch.float32
    model.train()
    model.to(device)
    model_return = None
    optim = Adam(params=model.parameters(), lr=1e-5)
    
    T = 0
    eps = eps_start
    best_achieved = -1000000
    
    while n_done < episodes:
        states_buffer = torch.empty(size=(1, 2), dtype=torch.float32)
        actions_buffer = torch.tensor([], dtype=torch.uint8)
        rewards_buffer = torch.tensor([], dtype=torch.float32)
        next_states_buffer = torch.empty(size=(1, 2), dtype=torch.float32)
        
        total_reward = 0
        
        state, info = env.reset()
        state = process_observation(state)
        state = state.to(device)

        with torch.no_grad():
            action = model.select_action(state, eps=eps)

        for t in count(0, 1):
            T += 1
            if T > target_model_update:
                with torch.no_grad():
                    model_target = copy.deepcopy(model).to(device)
                    T = 0
                
            next_state, reward, terminated, done, _info = env.step(action)
            next_state = process_observation(next_state).to(device)

            if terminated:
                loss = compute_q_loss_terminal(model, next_state, torch.tensor([action], dtype=torch.uint8), torch.tensor([reward], dtype=torch.float32), alpha, device)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 2)
                optim.step()
                
                history += [total_reward]
                if n_done % 2 == 0:
                    print("episode:", n_done, 'completed by:', t, 'reward:', total_reward)
                eps = np.clip(eps - 1/episodes, 0.05, 1)
                n_done += 1
                if total_reward > best_achieved:
                    with torch.no_grad():
                        model_return = copy.deepcopy(model)
                    best_achieved = total_reward
                break
                            
            states_buffer = torch.cat((states_buffer, state.to('cpu')))
            actions_buffer = torch.cat((actions_buffer, torch.tensor([action], dtype=torch.uint8)))
            rewards_buffer = torch.cat((rewards_buffer, torch.tensor([reward], dtype=torch.float32)))
            next_states_buffer = torch.cat((next_states_buffer, next_state.to('cpu')))
                        
            total_reward += reward
            
            with torch.no_grad():
                next_action = model.select_action(next_state, eps=eps)

            state = next_state
            action = next_action
            
            loss = compute_q_loss(model, model_target, states_buffer[-1], actions_buffer[-1], rewards_buffer[-1], next_states_buffer[-1], gamma, alpha, device)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2)
            optim.step()
            
            if (n_done % 2 == 0) and t==100:
                with torch.no_grad():
                    print('q_values:', model(next_state), 'loss:', loss)
            
            if random.random() < 0.05 and len(states_buffer) > batch_size:
                # there seems to be a bug with using state with index 0, no idea why
                samples = np.random.randint(2, len(states_buffer) - 1, size=batch_size)
                loss = torch.mean(compute_q_loss(model, model_target, states_buffer[samples], actions_buffer[samples], rewards_buffer[samples], next_states_buffer[samples], gamma, alpha, device))
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 2)
                optim.step()
                
    save_checkpoint(model_return)
    model_return.eval()
    return model_return, history
