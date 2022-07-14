import os
import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip

ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME).unwrapped

N_ACTIONS = env.action_space.n
STATE_SHAPE = env.observation_space.sample().shape

EPISODES = 1000 
EPS_START = 0.9 
EPS_END = 0.05  
EPS_DECAY = 200 
GAMMA = 0.75 
LR = 0.001 
HIDDEN_LAYER = 164 
BATCH_SIZE = 64  
MAX_STEPS = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

model = Network().to(device)
target = Network().to(device)

memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)

steps_done = 0
ed = []

def plot_durations(d):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(d)

    plt.savefig('episode_durations')

def botPlay():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    state = env.reset()
    steps = 0
    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = select_action(FloatTensor(np.array([state])))
        next_state, reward, done, _ = env.step(action.item())

        state = next_state
        steps += 1

        if done or steps >= MAX_STEPS:
            break

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_gif('CartPole.gif', fps=20)

def select_action(state, train=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if train:
        if sample > eps_threshold:
            with torch.no_grad():
                return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(2)]])
    else:
        with torch.no_grad():
            return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)

def run_episode(episode, env):
    state = env.reset()
    steps = 0
    while True:
        action = select_action(FloatTensor(np.array([state])))
        next_state, reward, done, _ = env.step(action.item())

        if done:
            if steps < 30:
                reward -= 10
            else:
                reward = -1
        if steps > 100:
            reward += 1
        if steps > 200:
            reward += 1
        if steps > 300:
            reward += 1

        memory.push((FloatTensor(np.array([state])),
                     action,  # action is already a tensor
                     FloatTensor(np.array([next_state])),
                     FloatTensor(np.array([reward]))))

        learn()

        state = next_state
        steps += 1

        if done or steps >= MAX_STEPS:
            ed.append(steps)
            print("[Episode {:>5}]  steps: {:>5}".format(episode, steps))
            if sum(ed[-10:])/10 > 800:
                return True
            break
    return False

def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = target(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for e in range(EPISODES):
    complete = run_episode(e, env)

    if complete:
        print('complete...!')
        break

    if (e+1) % 5 == 0:
        mp = list(target.parameters())
        mcp = list(model.parameters())
        n = len(mp)
        for i in range(0, n):
            mp[i].data[:] = mcp[i].data[:]

# plot_durations(ed)
botPlay()