import random
import gym
from gym.envs.classic_control import MountainCarEnv
import numpy as np


class Environment(object):
    def __init__(self, env):
        self._env = env

    @property
    def n_state(self):
        return len(self.features(self._env.state))

    @property
    def n_action(self):
        return self._env.action_space.n

    def features(self, state):
        return np.array(list(state) + [state[0]**2] + [state[1]**2] + [state[0]*state[1]] + [1]).reshape(-1,1)

    def reset(self):
        state = self._env.reset()
        return self.features(state)

    def step(self, action):
        gym_action = np.argmax(action)
        state, reward, is_done, _ = self._env.step(gym_action)
        return self.features(state), reward, is_done, _


class Agent(object):

    def __init__(self, n_state, n_action, alpha = .1, gamma = 0.9999, epsilon = 0.1, lmbda = .9):
        self._n_state = n_state
        self._n_action = n_action
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._lmbda = lmbda
        self._W = np.zeros((self._n_state, self._n_action))

        self.reset()

    def reset(self):
        self._z = np.zeros_like(self._W)
        self._Q_old = 0.

    def q(self, state, action):
        return (state.T @ self._W @ action)

    def select_action(self, state):
        r = random.uniform(0,1)
        if r <= (1-self._epsilon):
            index = np.argmax((state.T @ self._W)[0])
        else:
            index = random.randint(0,2)
        action = np.zeros((self._n_action,))
        action[index] = 1.0
        action = action.reshape((-1,1))
        return action

    def make_x(self, state, action):
        x = np.zeros((self._n_state, self._n_action))
        x[:,np.argmax(action)] = state[:,0]
        return x

    def update(self, state, action, reward, next_state, next_action):
        Q = self.q(state, action)
        Q_new = self.q(next_state, next_action)
        delta = reward + self._gamma*Q_new - Q
        x = self.make_x(state, action)
        self._z = self._gamma * self._lmbda * self._z + \
                  (1 - self._alpha * self._gamma * self._lmbda * (state.T @ self._z @ action)[0]) * x
        self._W = self._W + self._alpha * (delta + Q - self._Q_old) * self._z - \
                  self._alpha * (Q - self._Q_old) * x
        self._Q_old = Q_new


gym_env = MountainCarEnv()
env = Environment(gym_env)
agent = Agent(env.n_state, env.n_action)
episodes = 100

for i in range(episodes):
    agent.reset()

    state = env.reset()
    action = agent.select_action(state)
    is_done = False
    cumulative_reward = 0
    iterations = 0

    while not is_done:
        new_state, reward, is_done, _ = env.step(action)
        new_action = agent.select_action(new_state)
        agent.update(state, action, reward, new_state, new_action)
        state = new_state
        action = new_action

        iterations += 1
        cumulative_reward += reward

    print("Episode %s, iterations: %s, cumulative_reward: %s" % (i, iterations, cumulative_reward))

