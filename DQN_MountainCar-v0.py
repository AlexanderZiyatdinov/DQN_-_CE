import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

INFO_MESSAGE = "Episode {a}/ {b}. Done: {c} steps. Reward: {d}"
LEARNING_RATE = 0.001
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEM_SIZE = 50000
TRAINING_DATA_SIZE = 10000
GAMMA = 0.95  # Коэффициент дисконтирования


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEM_SIZE)
        self.epsilon = 1.0  # вероятность исследования среды
        self.model = self.build_model()

    def act(self, state):
        return (random.randrange(self.action_size)
                if random.uniform(0, 1) < self.epsilon
                else np.argmax(self.model.predict(state)[0]))

    def build_model(self):
        """Построение модели с помощью Keras"""
        model = Sequential()
        # Функция апроксимации - RELU
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def play_with_sample(self):
        if TRAINING_DATA_SIZE >= len(self.memory):
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        state = np.zeros((BATCH_SIZE, self.state_size))
        next_state = np.zeros((BATCH_SIZE, self.state_size))
        action, reward, done = [], [], []

        for i in range(BATCH_SIZE):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.model.predict(state)
        target_next = self.model(next_state)

        # Если состояние терминальное, то возвращаем R
        # Иначе Уравнение Беллмана
        for i in range(len(batch)):
            Bellman_equation = reward[i] + GAMMA * np.amax(target_next[i])
            target[i][action[i]] = reward[i] if done[i] else Bellman_equation

        self.model.fit(np.array(state), np.array(target), batch_size=BATCH_SIZE,
                       verbose=0)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def print_msg(*args):
    a, b, c, d = args[0], args[1], args[2], args[3]
    print(INFO_MESSAGE.format(a=a + 1, b=b, c=c, d=d))


if __name__ == "__main__":
    print_msg(1, 2, 3, 4)
    # Задаем среду
    env = gym.make('MountainCar-v0')

    # Задаем константы
    total_episodes = 750
    total_steps = env.spec.max_episode_steps
    actions = env.action_space.n
    states = env.observation_space.shape[0]
    max_reward = 0

    # Создаем агента
    agent = DQNAgent(state_size=states, action_size=actions)

    for episode in range(total_episodes):
        total_reward = 0
        state = np.reshape(env.reset(), [1, states])

        for ep_step in range(total_steps):
            env.render()

            action = agent.act(state)
            observation, reward, done, _ = env.step(action)

            if observation[1] > state[0][1] >= 0 and observation[1] >= 0:
                reward = 20
            if observation[1] < state[0][1] <= 0 and observation[1] <= 0:
                reward = 20
            if done and ep_step < total_steps - 1:
                reward += 10000
            else:
                reward -= 25

            total_reward += reward

            next_state = np.reshape(observation, [1, states])
            data = state, action, reward, next_state, done
            agent.memory.append(data)
            state = next_state
            agent.play_with_sample()

            if done:
                print_msg(episode + 1, total_episodes, ep_step, total_reward)
                break

            elif ep_step >= total_steps - 1:
                print_msg(episode + 1, total_episodes, ep_step, total_reward)

        if total_reward >= max_reward:
            max_reward = total_reward

    env.close()
