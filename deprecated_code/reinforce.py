import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import Model
from keras import models
from keras.utils import to_categorical
import keras.backend as K
import gym
from gym.wrappers import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = 1000


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, env, lr):
        '''
        Function to instantiate REINFORCE alorithm implementation class

        Parameters:
        model: Parsed model from JSON file
        env: Gym environment instance(Lunar-lander)
        lr: Learning rate for optimizer
        '''
        self.nstate = len(env.observation_space.high)
        self.naction = env.action_space.n
        self.env = env
        state_input = layers.Input((self.nstate, ))
        gt = layers.Input((1, ))

        cur = state_input
        for layer in model.layers:
            cur = layer(cur)
        func_model = Model(state_input, cur)
        y = func_model(state_input)

        self.model = Model([state_input, gt], y)

        adam = keras.optimizers.Adam(lr=lr)

        self.model.compile(optimizer=adam, loss=self.reinforce_loss(gt))
        self.t = T
        self.episode = 50000
        self.test_episode = 100

    def reinforce_loss(self, gt):
        '''
        Function to define loss function for the algorithm

        Parameters:
        gt: Future discounted returns

        Output:
        loss function for REINFORCE algorithm
        '''
        def f(y_true, y_pred):
            sum = K.log(K.sum(y_pred * y_true, axis=1))
            l = - K.mean(gt * sum, keepdims=False)
            return l
        return f

    def train(self, gamma=1.0):
        '''
        Function to train the network

        Parameters:
        gamma: Discount factor
        '''
        states, actions, rewards = self.generate_episode()
        len = states.shape[0]
        gt = []
        for t in range(len):
            g = 0.0
            for k in range(t, len):
                g += gamma**(k-t)*rewards[k]
            gt.append(g)
        gt = np.array(gt)

        actions = to_categorical(actions, self.naction)
        self.model.fit([states, gt], actions, verbose=False)

    def test(self):
        '''
        Function to evaluate performance of the current policy network

        Parameters:
        None

        Output:
        Maximum, mean and standard deviation of the test reward
        '''
        total_reward = []
        total_step = []
        for e in range(self.test_episode):
            is_terminal = False
            i = 0
            state = self.env.reset()
            cum = 0.0
            while i < self.t and not is_terminal:
                state = np.reshape(state, (1, -1))
                action = self.model.predict([state, np.empty((1, 1))])
                action = np.argmax(action)
                next_state, reward, is_terminal, _ = self.env.step(action)

                i += 1
                cum += reward
                state = next_state
            total_step.append(i)
            total_reward.append(cum)
        return np.max(total_reward), np.mean(total_reward), np.std(total_reward), np.mean(total_step)

    def run(self, path):
        '''
        Function to run the training and inference as well as  saving checkpoints

        Parameters:
        path: Path to the folder to save model file and rewards plot
        '''
        y = []
        ystd = []
        best_reward = -1000
        converge = 0
        for i in range(self.episode):
            self.train()

            if i % 100 == 0:
                stats = self.test()
                if stats[1] >= 200:
                    converge += 1
                else:
                    converge = 0
                y.append(stats[1])
                ystd.append(stats[2])
                print('test reward', i, stats)
                if stats[1] > best_reward:
                    best_reward = stats[1]
                    self.model.save(path + str(best_reward) + 'checkpoint.h5')

                x = range(len(y))
                plt.errorbar(x, y, yerr=ystd)
                plt.xlabel('10^2 episodes')
                plt.title('REINFORCE Training')
                plt.savefig(path + 'train_process'+str(i)+'.png')

                if converge >= 30:
                    break

    def generate_episode(self, render=False):
        '''
        Function to generate transition tuples for the environment

        Parameters:
        render: Argument to specify whether video is to be rendered

        Output:
        Transition tuples for the episode
        '''
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        is_terminal = False
        i = 0
        while i < self.t and not is_terminal:
            state = np.reshape(state, (1, -1))
            action = self.model.predict([state, np.empty((1, 1))])

            action = np.random.choice(self.naction, size=1, p=action[0])[0]
            next_state, reward, is_terminal, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            i += 1
        states = np.array(states)
        states = np.reshape(states, (states.shape[0], -1))
        actions = np.array(actions)
        rewards = np.array(rewards)

        return states, actions, rewards


def parse_arguments():
    '''
    Function to parse arguments as per argparse

    Parameters:
    None

    Output:
    Parsed arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    print('action', env.action_space)
    nstate = len(env.observation_space.low)
    env.reset()
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    reinforce = Reinforce(model, env, lr=5e-4)
    # reinforce.train()
    reinforce.run(path='trained/')


if __name__ == '__main__':
    main(sys.argv)
