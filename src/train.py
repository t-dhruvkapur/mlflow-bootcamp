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

import yaml

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME).unwrapped

N_ACTIONS = env.action_space.n
STATE_SHAPE = env.observation_space.sample().shape

# Tag Logging (one at a time)
mlflow.set_tag("Environment Name", ENV_NAME)
mlflow.set_tag("Number of Actions", N_ACTIONS)
mlflow.set_tag("State Shape", STATE_SHAPE)

EPISODES = 1000 
EPS_START = 0.9 
EPS_END = 0.05  
EPS_DECAY = 200 
GAMMA = 0.75 
LR = 0.001 
HIDDEN_LAYER = 164 
BATCH_SIZE = 64  
MAX_STEPS = 1000

# Parameter Logging (multiple at once)
mlflow.log_params({
    "Episodes": EPISODES,
    "Epsilon Start": EPS_START,
    "Epsilon End": EPS_END,
    "Epsilon Decay": EPS_DECAY,
    "Discount Factor": GAMMA,
    "Learning Rate": LR,
    "Batch Size": BATCH_SIZE,
    "Max Steps Per Episode": MAX_STEPS
})


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlflow.set_tag("Device", device)

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

    # Log Artifact
    mlflow.log_artifact(local_path='CartPole.gif', artifact_path='BotPlay')

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
    ep_loss = 0
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

        step_loss = learn()

        if step_loss is not None:
            ep_loss += step_loss

        state = next_state
        steps += 1

        if done or steps >= MAX_STEPS:
            ed.append(steps)

            # Logged metrics (time-series)
            mlflow.log_metric("Episodic Loss", ep_loss / steps)
            mlflow.log_metric("Episode Duration", steps)

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

    return loss.item()

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


# Logging scalar metric
mlflow.log_metric('Mean Duration', sum(ed)/len(ed))

botPlay()

# Save the model as an artifact
torch.save(model, "model.pt")
mlflow.log_artifact("model.pt", "artifact_model")

# Saving the model as an MLmodel
# Load environment
with open('conda.yml', 'r') as f:
    conda_env = yaml.safe_load(f)

# Create signature
input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, *STATE_SHAPE))])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, N_ACTIONS))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log Model as MLModel
mlflow.pytorch.log_model(model, 'ml_model', conda_env, signature=signature)

# Register Model
mlflow.register_model(f'runs:/{mlflow.active_run().info.run_id}/ml_model', 'cartpole-dqn')
