import gym
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, ReLU, Linear, Softmax, CrossEntropyLoss
from torch.optim import Adam
from numpy.random import choice

EPSILON = 0.001
COUNT_OF_EPOCHS = 30
COUNT_OF_SESSIONS = 30
STEPS_IN_SESSION = 6000
Q_PARAMETER = 0.92
LEARNING_RATE = 0.01
HIDDEN_LAYERS_DIM = 100


class CrossEntropyAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = Sequential(Linear(state_size, HIDDEN_LAYERS_DIM), ReLU(),
                                  Linear(HIDDEN_LAYERS_DIM, action_size), )
        self.softmax = Softmax()
        self.loss_function = CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=LEARNING_RATE)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.network(state)
        action_prob = self.softmax(action).detach().numpy()
        return choice(len(action_prob), p=action_prob)

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session['states'])
            elite_actions.extend(session['actions'])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss_function(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def get_session(env, agent):
    session = {}
    states, actions = [], []
    total_reward = 0

    state = env.reset()

    for session_step in range(STEPS_IN_SESSION):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)
        env.render()

        state, reward, done, _ = env.step(action)
        total_reward += reward
        pos, _ = state

        if abs(pos - 0.5) < EPSILON:
            print(reward, session_step)
            total_reward += 5
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions):
    elite_sessions = []
    total_rewards = np.array([session['total_reward'] for session in sessions])

    for session in sessions:
        if (session['total_reward'] > np.quantile(total_rewards, Q_PARAMETER)
                or session['total_reward']) > 0:
            elite_sessions.append(session)

    return elite_sessions


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    rewards = []

    obs = env.observation_space.shape[0]
    space = env.action_space.n
    agent = CrossEntropyAgent(obs, space)

    for episode in range(COUNT_OF_EPOCHS):
        sessions = [get_session(env, agent) for i in
                    range(COUNT_OF_SESSIONS)]

        total_reward = np.mean([sess['total_reward'] for sess in sessions])
        elite_sessions = get_elite_sessions(sessions)

        if len(elite_sessions) > 0:
            agent.update_policy(elite_sessions)

        print(f"Episode: {episode + 1}. Reward: {total_reward}")
        rewards.append(total_reward)
