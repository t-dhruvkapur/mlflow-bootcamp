import gym

env = gym.make('CartPole-v1')

state = env.reset()
for i in range(300):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

env.close()
