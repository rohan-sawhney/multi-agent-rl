from __future__ import division
import gym
import argparse
import copy

import numpy as np
# import tensorflow as tf
import itertools
import time
import os
import pickle as pk
from collections import namedtuple
from collections import deque
import code

from make_env import make_env

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class Agent(object):

    def __init__(self):
        self.training = True
        self.step = 1
        self.epoch = 1


def fill_memory(options, env, memories):
    for i in range(20):
        state = env.reset()
        for step in range(50):
            if options.render:
                env.render()
            actions = []
            onehot_actions = []
            for i in range(env.n):
                if i == env.n - 1:
                    role = 1
                else:
                    role = 0
                action = np.random.randint(env.action_space[i].n)
                actions.append(action)
                onehot_action = np.zeros(env.action_space[i].n)
                onehot_action[action] = options.movement_rate
                onehot_actions.append(onehot_action)

            next_state, reward, done, info = env.step(onehot_actions)
            # reward = np.clip(reward, -1., 1.)

            for i in range(env.n):
                if i == env.n - 1:
                    role = 1
                else:
                    role = 0
                memories[role].append(state[i], actions[i], reward[i], done[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str)
    parser.add_argument('--folder', default='', type=str)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--movement_rate', default=.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_episodes', default=1e6, type=int)
    parser.add_argument('--copy_interval', default=1e4, type=int)
    parser.add_argument('--linear_size', default=50, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--window_length', default=1, type=int)
    parser.add_argument('--stats_window', default=100, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    options = parser.parse_args()
    if options.folder == "":
        options.folder = options.env
    if not os.path.isdir(options.folder):
        os.makedirs(options.folder)

    env = make_env(options.env, options.benchmark)
    np.random.seed(123)
    env.seed(123)

    # TODO: This is the code for separate DQN for each agent, need to tweak
    # Keras DQN
    n_actions = [env.action_space[0].n, env.action_space[-1].n]
    n_states = [env.observation_space[0].shape[0],
                env.observation_space[-1].shape[0]]
    filename = options.folder + "/agent.pk"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            agent = pk.load(f)
    else:
        agent = Agent()

    models = []
    target_models = []
    memories = []
    policies = []
    for n in range(len(n_actions)):
        filename = options.folder + "/model_" + str(n) + ".h5"
        if os.path.isfile(filename):
            model = load_model(filename)
            target_model = load_model(filename)
        else:
            model = Sequential()
            model.add(Dense(options.linear_size, activation='relu',
                            input_shape=(n_states[n],)))
            model.add(Dense(options.linear_size, activation='relu'))
            model.add(Dense(n_actions[n], activation='linear'))
            model.compile(optimizer=Adam(
                lr=options.learning_rate), loss='mse', metrics=['mae'])
            target_model = Sequential()
            target_model.add(Dense(options.linear_size, activation='relu',
                                   input_shape=(n_states[n],)))
            target_model.add(Dense(options.linear_size, activation='relu'))
            target_model.add(Dense(n_actions[n], activation='linear'))
            target_model.compile(optimizer=Adam(
                lr=options.learning_rate), loss='mse', metrics=['mae'])
            print(model.summary())
        models.append(model)
        target_models.append(model)
        memories.append(SequentialMemory(
            limit=options.memory_size, window_length=options.window_length))
        policies.append(LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                             value_max=1., value_min=.1,
                                             value_test=.05,
                                             nb_steps=100000))
        policies[-1]._set_agent(agent)

    fill_memory(options, env, memories)

    epoch = agent.epoch
    last_epoch_step = 0
    state = env.reset()
    tot_reward = np.zeros(env.n)
    reward_window = deque(maxlen=options.stats_window)
    loss_window = deque(maxlen=options.stats_window * len(n_actions))
    my_history = []
    filename = options.folder + "/history.pk"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            my_history = pk.load(f)

    for step in itertools.count(agent.step):
        agent.step = step
        agent.epoch = epoch
        if epoch >= options.train_episodes:
            break
        if options.render:
            env.render()
        if step % options.copy_interval == 0:
            for i in range(len(models)):
                target_models[i].set_weights(models[i].get_weights())
        actions = []
        onehot_actions = []
        for i in range(env.n):
            if i == env.n - 1:
                role = 1
            else:
                role = 0
            # action.append(np.random.random(4 + env.world.dim_c))
            q_value = models[role].predict(state[i].reshape(1, -1))
            action = policies[role].select_action(q_values=q_value[0])
            actions.append(action)
            onehot_action = np.zeros(5 + env.world.dim_c)
            onehot_action[action] = options.movement_rate
            onehot_actions.append(onehot_action)

        next_state, reward, done, info = env.step(onehot_actions)
        # done = [any(reward[:3])] * 4
        # reward = np.clip(reward, -1., 1.)
        tot_reward += np.array(reward)

        for i in range(env.n):
            if i == env.n - 1:
                role = 1
            else:
                role = 0
            memories[role].append(state[i], actions[i], reward[i], done[i])

        losses = []
        for role in range(len(n_actions)):
            experiences = memories[role].sample(options.batch_size)

            # Start by extracting the necessary parameters (we use a
            # vectorized implementation).
            state0_batch = []
            state1_batch = []
            reward_batch = []
            action_batch = []
            for e in experiences:
                state0_batch.append(e.state0[0])
                state1_batch.append(e.state1[0])
                reward_batch.append(e.reward)
                action_batch.append(e.action)

            # Prepare and validate parameters.
            state0_batch = np.array(state0_batch)
            state1_batch = np.array(state1_batch)
            reward_batch = np.array(reward_batch)

            target_q_values = target_models[role].predict(
                state1_batch, batch_size=options.batch_size)
            q_batch = np.max(target_q_values, axis=1).flatten()
            target_q_values = target_models[role].predict(
                state0_batch, batch_size=options.batch_size)

            discounted_reward_batch = options.gamma * q_batch
            Rs = reward_batch + discounted_reward_batch
            target_q_values[range(len(action_batch)), action] = Rs
            # for idx, (target, R, action) in enumerate(zip(target_q_values, Rs, action_batch)):
            #     target[action] = R

            history = models[role].fit(state0_batch, target_q_values,
                                       batch_size=options.batch_size, verbose=0)
            losses.append(np.mean(history.history['loss']))
            loss_window.append(losses)

        # if step % 100 == 0:
        #     print(epoch, step, losses)
        if step % 10000 == 0:
            for n in range(len(n_actions)):
                filename = options.folder + "/model_" + str(n) + ".h5"
                models[n].save(filename)
            filename = options.folder + "/agent.pk"
            with open(filename, 'wb') as f:
                pk.dump(agent, f)
            filename = options.folder + "/history.pk"
            with open(filename, 'wb') as f:
                pk.dump(my_history, f)

        if any(done) or step - last_epoch_step > 100:
            state = env.reset()
            epoch += 1
            last_epoch_step = step
            reward_window.append(np.sum(tot_reward))
            my_history.append([np.mean(loss_window), np.mean(reward_window)])
            if epoch % 10 == 0:
                print(epoch, step, my_history[-1])
            tot_reward = np.zeros(env.n)


if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
