import numpy as np
import tensorflow as tf
import pathlib
import general_utilities


class Actor:

    def __init__(self, scope, session, n_actions, action_bound,
                 eval_states, target_states, learning_rate=0.001, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.eval_states = eval_states
        self.target_states = target_states
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.eval_actions = self.build_network(self.eval_states,
                                                   scope='eval', trainable=True)
            self.target_actions = self.build_network(self.target_states,
                                                     scope='target', trainable=False)

            self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=scope + '/eval')
            self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=scope + '/target')

            self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                                  for t, e in zip(self.target_weights, self.eval_weights)]

    def build_network(self, x, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)
            h1 = tf.layers.dense(x, 50, activation=tf.nn.relu,
                                 kernel_initializer=W, bias_initializer=b,
                                 name='h1', trainable=trainable)
            actions = tf.layers.dense(h1, self.n_actions, activation=tf.nn.tanh,
                                      kernel_initializer=W, bias_initializer=b,
                                      name='actions', trainable=trainable)
            scaled_actions = tf.multiply(actions, self.action_bound,
                                         name='scaled_actions')

        return scaled_actions

    def add_gradients(self, action_gradients):
        with tf.variable_scope(self.scope):
            self.action_gradients = tf.gradients(ys=self.eval_actions,
                                                 xs=self.eval_weights,
                                                 grad_ys=action_gradients)
            optimizer = tf.train.AdamOptimizer(-self.learning_rate)
            self.optimize = optimizer.apply_gradients(zip(self.action_gradients,
                                                          self.eval_weights))

    def learn(self, actors, states):
        a = {}
        for i in range(len(states)):
            a[actors[i].eval_states] = states[i]

        self.session.run(self.optimize, feed_dict={**a})
        self.session.run(self.update_target)

    def choose_action(self, state):
        return self.session.run(self.eval_actions,
                                feed_dict={self.eval_states: state[np.newaxis, :]})[0]


class Critic:

    def __init__(self, scope, session, n_actions, actors_eval_actions,
                 actors_target_actions, eval_states, target_states,
                 rewards, learning_rate=0.001, gamma=0.9, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.actors_eval_actions = actors_eval_actions
        self.actors_target_actions = actors_target_actions
        self.eval_states = eval_states
        self.target_states = target_states
        self.rewards = rewards

        with tf.variable_scope(scope):
            self.eval_values = self.build_network(self.eval_states,
                                                  self.actors_eval_actions,
                                                  'eval', trainable=True)
            self.target_values = self.build_network(self.target_states,
                                                    self.actors_target_actions,
                                                    'target', trainable=False)

            self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=scope + '/eval')
            self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=scope + '/target')

            self.target = self.rewards + gamma * self.target_values
            self.loss = tf.reduce_mean(tf.squared_difference(self.target,
                                                             self.eval_values))

            self.optimize = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss)
            self.action_gradients = []
            for i in range(len(self.actors_eval_actions)):
                self.action_gradients.append(tf.gradients(ys=self.eval_values,
                                                          xs=self.actors_eval_actions[i])[0])

            self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                                  for t, e in zip(self.target_weights, self.eval_weights)]

    def build_network(self, x1, x2, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)

            first = True
            for i in range(len(x1)):
                h1 = tf.layers.dense(x1[i], 50, activation=tf.nn.relu,
                                     kernel_initializer=W, bias_initializer=b,
                                     name='h1-' + str(i), trainable=trainable)
                h21 = tf.get_variable('h21-' + str(i), [50, 50],
                                      initializer=W, trainable=trainable)
                h22 = tf.get_variable('h22-' + str(i), [self.n_actions[i], 50],
                                      initializer=W, trainable=trainable)

                if first == True:
                    h3 = tf.matmul(h1, h21) + tf.matmul(x2[i], h22)
                    first = False
                else:
                    h3 = h3 + tf.matmul(h1, h21) + tf.matmul(x2[i], h22)

            b2 = tf.get_variable('b2', [1, 50], initializer=b,
                                 trainable=trainable)
            h3 = tf.nn.relu(h3 + b2)
            values = tf.layers.dense(h3, 1, kernel_initializer=W,
                                     bias_initializer=b, name='values',
                                     trainable=trainable)

        return values

    def learn(self, states, actions, rewards, states_next):
        s = {i: d for i, d in zip(self.eval_states, states)}
        a = {i: d for i, d in zip(self.actors_eval_actions, actions)}
        sn = {i: d for i, d in zip(self.target_states, states_next)}

        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={**s, **a, **sn,
                                                                          self.rewards: rewards})
        self.session.run(self.update_target)
        return loss
