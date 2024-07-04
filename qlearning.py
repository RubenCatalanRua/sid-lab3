import gymnasium as gym
import numpy as np
import random
import time
import pandas as pd
import sys


from solution_concepts import SolutionConcept
from game_model import GameModel
import abc


class QLearningAgent:
    def __init__(
        self,
        env: GameModel,
        gamma: float,
        learning_rate: float,
        epsilon: float,
        agent_id: int,
        seed = 1,
    ):
        self.env = env
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rand = random.Random(seed)     # Any would work, honestly
        self.td_error = {"TD_err": []}
        self.id= agent_id


    # Q-Learning code, adapted for this game
    def select_action(self, state, training=True):
        if training and self.rand.random() <= self.epsilon:
            return self.rand.choice(range(self.env.num_actions))
        else:
            return np.argmax(self.Q[state,])

    # One change needed, that is, adding the TD error for metric purposes
    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error
        self.td_error["TD_err"].append(td_target)

    def learn_from_episode(self, actions, rewards, state, next_state):
        total_reward = 0
        action = self.select_action(state)
        reward = rewards[self.id]

        self.update_Q(state, action, reward, state)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon