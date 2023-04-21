# Spring 2023, 535515 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_3")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.fc1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        x = self.fc1(torch.cat([inputs, actions], dim=1))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        #mu = self.actor((Variable(state)))
        #mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 

        with torch.no_grad():
            if action_noise != None:
                action = self.actor(state.to(device))
                action = action + action_noise.to(device)
            else:
                action = self.actor(state.to(device))

        action = torch.clamp(action, min=-2.0, max=2.0)
        return action

        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        #state_batch = Variable(torch.cat(batch.state))
        #action_batch = Variable(torch.cat(batch.action))
        #reward_batch = Variable(torch.cat(batch.reward))
        #next_state_batch = Variable(torch.cat(batch.next_state))
        #done_batch = Variable(torch.cat(batch.done))

        state_batch = Variable(torch.cat([trans.state for trans in batch])).to(device)
        action_batch = Variable(torch.cat([trans.action for trans in batch])).to(device)
        reward_batch = Variable(torch.cat([trans.reward for trans in batch])).view(-1, 1).to(device)
        next_state_batch = Variable(torch.cat([trans.next_state for trans in batch])).to(device)
        done_batch = Variable(torch.cat([trans.done for trans in batch])).view(-1, 1).to(device)

        #print(state_batch.shape)
        #print(action_batch.shape)
        #print(reward_batch.shape)
        #print(next_state_batch.shape)
        #print(done_batch.shape)

        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic

        ## update critic ##
        # critic loss
        q_value = self.critic(state_batch, action_batch)
        with torch.no_grad():
            a_next = self.actor_target(next_state_batch)
            q_next = self.critic_target(next_state_batch, a_next)
            q_target = reward_batch + self.gamma * q_next * (1 - done_batch)  # done == 1: final state
            
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)
        
        # optimize critic
        self.actor.zero_grad()
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        ## update actor ##
        # actor loss
        action = self.actor(state_batch)
        actor_loss = (self.critic(state_batch, action).mean()) * -1  # gradient ascend
        
        # optimize actor
        self.actor.zero_grad()
        self.critic.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():    
    num_episodes = 200
    gamma = 0.995
    tau = 0.002
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 128
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0
    env_name = 'Pendulum-v0'
    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.Tensor([env.reset()])

        episode_reward = 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            noise = ounoise.noise()
            noise = torch.FloatTensor(noise)
            action = agent.select_action(state, noise)
            next_state, reward, done, _ = env.step([action.item()])
            total_numsteps += 1

            memory.push(
                state,
                action,
                torch.Tensor([reward]),
                torch.Tensor(next_state).view(1, -1),
                torch.Tensor([int(done)]),
            )

            if total_numsteps % updates_per_step == 0 and batch_size <= len(memory):
                batch = memory.sample(batch_size)
                agent.update_parameters(batch)

            episode_reward += reward
            state = torch.Tensor(next_state).view(1, -1)

            if done:
                break
            ########## END OF YOUR CODE ########## 
            

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step([action.item()])
                
                env.render()
                
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            writer.add_scalar("Train/ep_reward", episode_reward, i_episode)
            writer.add_scalar("Train/ewma_reward", ewma_reward, i_episode)
            writer.add_scalar("Train/length", t, i_episode)
    
    agent.save_model(env_name, '.pth')        
 

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    #env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v0')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train()


