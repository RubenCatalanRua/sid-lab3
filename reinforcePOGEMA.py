import gymnasium as gym
import numpy as np
import random
import time
import pandas as pd
import sys


from game_model import GameModel
import abc


class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, t_max, agent_id, lr_decay=1, seed=1):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.policy_table = np.ones(
            (self.env.num_states, self.env.num_actions))
        np.random.seed(seed)
        self.rand = random.Random(seed)
        self.t_max = t_max
        self.id = agent_id

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training and self.rand.random() <= action_probabilities.all():
            # Escogemos la acción según el vector de policy_table correspondiente a la acción,
            # con una distribución de probabilidad igual a los valores actuales de este vector
            return self.rand.choice(range(self.env.num_actions))
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        loss = - \
            np.sum(
                np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)
        policy_logits = np.log(self.policy_table)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]
                          ] += self.learning_rate * policy_gradient
            # Alternativa:
            # policy_gradient = 1.0 / action_probs[actions[t]]
            # policy_logits[states[t], actions[t]] += self.learning_rate * G_t * policy_gradient
        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / \
            np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def learn_from_episode(self, actions, rewards, state, next_state):
        episode = []
        step = 0
        while step < self.t_max:
            action = self.select_action(state)
            reward = rewards[self.id]
            episode.append((state, action, reward))
            state = next_state
            step = step + 1
        loss = self.update_policy(zip(*episode))
        self.learning_rate = self.learning_rate * self.lr_decay