import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras.models import Model
from keras.layers import Dense,Input
from keras import optimizers
import keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        '''
        Function to instantiate Actor Critic alorithm implementation class

        Parameters:
        model: Actor model
        lr: Learning rate for the actor model
        critic_model: Critic model
        critic_lr: Learning rate for the critic model
        n: Value of N for N-step A2C
        '''
        self.model = model
        self.critic_model = critic_model
        self.n = n
        
        adam_a = optimizers.Adam(lr = lr)
        adam_c = optimizers.Adam(lr = critic_lr)
        
        self.model.compile(optimizer=adam_a, loss=keras.losses.categorical_crossentropy)
        self.critic_model.compile(optimizer=adam_c, loss=keras.losses.mean_squared_error)

    def generate_episode(self, env, model,render=False):
       '''
        Function to generate transition tuples for the environment

        Parameters:
        env: Gym environment instance
        model: Model to predict the value functions
        render: Argument to specify whether video is to be rendered

        Output:
        Transition tuples for the episode
        '''
       states = []
       actions = []
       rewards = []
       state = env.reset()
       is_terminal = False
       i = 0
       # max_steps = 300
       # while i < max_steps and not is_terminal:
       while not is_terminal:
           
           state = np.reshape(state, (1, -1))
           action = model.predict(state)
           action = np.random.choice(env.action_space.n, size=1, p=action[0])[0]
           next_state, reward, is_terminal, _ = env.step(action)

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


    def train(self, env, gamma=1.0):
        '''
        Function to train the network

        Parameters:
        gamma: Discount factor
        '''
        states, actions, rewards = self.generate_episode(env,self.model)
        rewards=(0.01)*rewards

        T=len(states)
        N=self.n
        
        target = np.zeros([T,env.action_space.n])
        R = np.zeros([T,1])
        for t in range(T-1,-1,-1):
            if (t+N) >= T:
                V_end = 0
            else:
                V_end = self.critic_model.predict(np.reshape(states[t+N],[1,env.observation_space.shape[0]]))
            sum_ = 0
            for k in range(0,N):
                if (t+k) < T:
                    r = rewards[t+k]
                else:
                    r=0
                sum_ = sum_ + ((gamma**k)*r)
            R[t] = ((gamma**N)*V_end) + sum_
            target[t][actions[t]] = R[t] - self.critic_model.predict(np.reshape(states[t],[1,env.observation_space.shape[0]]))
        
        self.model.fit(states,target,verbose=False)
        self.critic_model.fit(states,R,verbose=False)

    def test(self,env,render=False):
        '''
        Function to evaluate performance of the current policy network

        Parameters:
        None

        Output:
        Maximum, mean and standard deviation of the test reward
        '''
        total_reward = []
        total_step = []
        max_steps = 300
        for e in range(100):
            is_terminal = False
            i = 0
            state = env.reset()
            cum = 0.0
            # while i < max_steps and not is_terminal:
            while not is_terminal:
                state = np.reshape(state, (1, -1))
                action = self.model.predict(state)
                action = np.argmax(action)
                next_state, reward, is_terminal, _ = env.step(action)

                i += 1
                cum += reward
                state = next_state
            total_step.append(i)
            total_reward.append(cum)
        return np.max(total_reward), np.mean(total_reward), np.std(total_reward), np.mean(total_step)


    def run(self, env,num_episodes,path):
        '''
        Function to run the training and inference as well as  saving checkpoints

        Parameters:
        env: Gym environment instance
        num_episodes: NUmber of episodes for training
        path: Path to the folder to save model file and rewards plot
        '''
        print('work dir:', path)
        y = []
        ystd = []
        converge = 0
        best_reward = -1000
        for i in range(num_episodes):
            self.train(env)

            if i % 100 == 0:
                stats = self.test(env)
                if stats[1] >= 200:
                    converge += 1
                else:
                    converge = 0
                y.append(stats[1])
                ystd.append(stats[2])
                print('test reward', i, stats)
                if stats[1] > best_reward:
                    best_reward = stats[1]
                    self.model.save(path + str(best_reward) + 'policycheckpoint.h5')
                    self.critic_model.save(path + str(best_reward) + 'valuecheckpoint.h5')

                x = range(len(y))
                plt.errorbar(x, y, yerr=ystd)
                plt.xlabel('10^2 episodes')
                plt.title('A2C Training')
                plt.savefig(path + 'train_process'+str(i)+'.png')

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
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    # Create critic model
    input_layer = Input(shape=env.observation_space.shape)
    hidden_layer1 = Dense(64,activation='relu',kernel_initializer = 'VarianceScaling')(input_layer)
    hidden_layer2 = Dense(32,activation='relu',kernel_initializer = 'VarianceScaling')(hidden_layer1)
    hidden_layer3 = Dense(32,activation='relu',kernel_initializer = 'VarianceScaling')(hidden_layer2)
    output_layer = Dense(1,activation='linear')(hidden_layer3)
    critic_model = Model(inputs=input_layer, outputs=output_layer)

    a2C_obj =  A2C(model, lr, critic_model, critic_lr, n)
    a2C_obj.run(env,num_episodes,"./save/")

if __name__ == '__main__':
    main(sys.argv)
