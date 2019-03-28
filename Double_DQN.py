# This implementation is based on double DQN algorithm as described by Hasselt et al. (https://arxiv.org/abs/1509.06461)

#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, random
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt

class QNetwork():
    '''
    Class to define the network architecture for calculation of Q values for the possible actions given the current state of agent

    Attributes:
    environment_name (str): Currently two Gym environments are supported (CartPole and Mountain Car)
    '''

    def __init__(self, environment_name):
        '''
        Function to define network architecture and instantiate a network object

        Parameters:
        environment_name: Name of OpenAI Gym environment for which training is to be done (CartPole or MountainCar)
        '''
        env = gym.make(environment_name)
        
        if environment_name == 'CartPole-v0':
            learning_rate = 0.001
            neurons_H1 = 32
            neurons_H2 = 16
            neurons_H3 = 16
            input_layer = Input(shape=env.observation_space.shape)
            hidden_layer1 = Dense(neurons_H1,activation='relu',kernel_initializer = 'he_uniform')(input_layer)
            hidden_layer2 = Dense(neurons_H2,activation='relu',kernel_initializer = 'he_uniform')(hidden_layer1)
            hidden_layer3 = Dense(neurons_H3,activation='relu',kernel_initializer = 'he_uniform')(hidden_layer2)
            output_layer = Dense(env.action_space.n,activation='linear')(hidden_layer3)
        else:
            learning_rate = 0.0001
            neurons_H1 = 32
            neurons_H2 = 32
            neurons_H3 = 16
            input_layer = Input(shape=env.observation_space.shape)
            hidden_layer1 = Dense(neurons_H1,activation='relu',kernel_initializer = 'he_uniform')(input_layer)
            hidden_layer2 = Dense(neurons_H2,activation='relu',kernel_initializer = 'he_uniform')(hidden_layer1)
            hidden_layer3 = Dense(neurons_H3,activation='relu',kernel_initializer = 'he_uniform')(hidden_layer2)
            output_layer = Dense(env.action_space.n,activation='linear')(hidden_layer2)

        model = Model(inputs=input_layer, outputs=output_layer)

        adam = optimizers.Adam(lr = learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        
        self.model = model
        self.target_model = model
        
    def save_model_weights(self, suffix):
        '''
        Function to save model weights to file

        Parameters:
        suffix: File name for saving model weights
        '''
        self.model.save_weights(suffix)

    def load_model(self, model_file):
        '''
        Function to load a previously saved model

        Parameters:
        model_file: File name for previously saved model
        '''
        self.model = load_model(model_file)

    def load_model_weights(self,weight_file):
        '''
        Function to load model weights from a previously saved model weights file

        Parameters:
        weight_file: File name for previously saved model weights file
        '''
        self.model.load_weights(weight_file)

class Replay_Memory():
    '''
    Class to generate the replay memory buffer to store transition tuples

    Attributes:
    memory_size: Size of replay buffer
    burn_in: Number of transtion tuples for intialization
    '''

    def __init__(self, memory_size=50000, burn_in=10000):
        '''
        Function to generate replay memory buffer and instantiate memory buffer object
        Parameters:
        memory_size: Number of transition tuples to be stored in the replay memory buffer 
        burn_in: Number of transition tuples to be initialized with randomly initialized agent to facilitate beginning of training
        '''
        self.memory = []
        self.current = 0
        self.max = memory_size

    def sample_batch(self, batch_size=32):
        '''
        Function to sample a batch of transition tuples for training

        Parameters:
        batch_size: Size of transition tuples batch to be sampled for training

        Output:
        Randomly sampled batch of transition tuples
        '''
        return random.sample(self.memory,batch_size)

    def append(self, transition):
        '''
        Function to add new transition tuples to replay memory buffer

        Parameters:
        transition: Transition tuple to be added to the replay memory buffer
        '''
        # Appends transition to the memory.
        if self.current<self.max:
            self.memory.append(transition)
            self.current = self.current + 1
        else:
            self.memory[self.current%self.max] = transition
            self.current = self.current + 1

class DQN_Agent():
    '''
    Class to initialize agent and train the agent in the specified environment

    Attributes:
    environment_name (str): Currently two Gym environments are supported (CartPole and Mountain Car)
    render: Argument to specify whether video is to be rendered
    '''

    def __init__(self, environment_name, render=False):
        '''
        Function to initialize environment, replay memory buffer and network instance

        Parameters:
        environment_name (str): Currently two Gym environments are supported (CartPole and Mountain Car)
        render: Argument to specify whether video is to be rendered
        '''
        self.network = QNetwork(environment_name)
        self.memory = Replay_Memory()
        self.num_ep = 10000
        self.env = gym.make(environment_name)
        self.render_decision = render
        self.env_name = environment_name
        if environment_name == 'CartPole-v0':
            self.discount = 0.99
        else:
            self.discount = 1
    
    def epsilon_greedy_policy(self, q_values,epsilon):
        '''
        Function to implement epsilon-greedy policy for the agent

        Parameters:
        q_values: Q-values for the possible actions
        epsilon: Parameter to define exploratory action probability

        Output:
        action: Action selected by agent as per epsilon-greedy policy
        '''
        num = random.random()
        if(num<epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action

    def greedy_policy(self, q_values):
        '''
        Function to implement greedy policy for the agent

        Parameters:
        q_values: Q-values for the possible actions

        Output:
        Action selected by agent as per greedy policy corresponding to maximum Q-value
        '''
        return np.argmax(q_values)

    def train(self):
        '''
        Function to train the agent and save checkpoint videos

        Parameters:
        None

        Output:
        Reward obtained after test
        '''
        step = 0
        performance_reward = []
        if self.render_decision:
            self.env.render()
            def f(x):
                return x%3333==0
            self.env.render()
            path ='video_double_DQN_'+self.env_name+'/'
            self.env = gym.wrappers.Monitor(self.env, path, video_callable=f,force = True)
            self.env.reset()
            
        for episode in range(self.num_ep):
            total_reward = 0.0
            done = False
            state = self.env.reset()
            while not done:
#                Initialize q values
                q_values = self.network.model.predict(np.reshape(state,[1,self.env.observation_space.shape[0]]))
#                Obtain action using q values through epsilon-greedy policy
                if self.env_name=='CartPole-v0' and step < 1000000:
                    epsilon = 0.5-((step*(0.5-0.05))/1000000)
#                elif self.env_name!='CartPole-v0' and step < 5000000:
#                    epsilon = 0.5-((step*(0.5-0.05))/5000000)
                else:
                    epsilon = 0.05
                action = self.epsilon_greedy_policy(q_values,epsilon)
                
#                Execute action to obtain next state
                next_state, reward, done, _ = self.env.step(action)
                
#                Append the transition obtained due to action taken
                self.memory.append((state,action,reward,next_state,done))
#                Change state to next state
                state = next_state
#                Sample random batch from replay memory
                batch = self.memory.sample_batch()
#                Generate array to store batch
                x_train=np.zeros([32,self.env.observation_space.shape[0]])
                y_train=np.zeros([32,self.env.action_space.n])
#                Extract batch data into separate lists
                c_stat, act, rew,n_stat,check = zip(*batch)
#                Obtain target values by using current input state using model
                y = self.network.model.predict(np.reshape(c_stat,[32,self.env.observation_space.shape[0]]))
#                Obtain next state target values by using next state using model
                q = self.network.model.predict(np.reshape(n_stat,[32,self.env.observation_space.shape[0]]))
#                Obtain next state target values by using next state using target model
                q_target = self.network.target_model.predict(np.reshape(n_stat,[32,self.env.observation_space.shape[0]]))
#                Obtain action for which q value is maximum
                action_max = np.argmax(q,axis=1)
                
                for i in range(32):
                    if check[i]:
                        y[i,act[i]] = rew[i]
                    else:
                        y[i,act[i]] = rew[i] + self.discount*(q_target[i][action_max[i]])
#                Store data in batches
                y_train = y
                x_train = np.array(c_stat)
#                Train network based on batches generated
                self.network.model.fit(x_train, y_train, epochs = 1,batch_size=32,verbose=0)
#                Find reward obtained in each episode
                total_reward = total_reward + reward
                step = step + 1
                if step%1000==0:
                    self.network.target_model.set_weights(self.network.model.get_weights())
                    
#           Evaluate performance after every 200 episodes
            if episode%200==0:
                mean,_ = self.test(20)
                performance_reward.append(mean)
#                Save model after every 1000 episodes
               if episode%1000==0:
                   name = 'double_DQN_model_'+self.env_name+'_'+str(episode)+'.h5'
                   self.network.model.save(name)
            if total_reward!=-200 or episode%100==0:
                print("After {} episodes, the reward in this episode is {}".format(episode,total_reward))
        return performance_reward
        

    def test(self, ep,model_file=None):
        '''
        Function to test the current policy

        Parameters:
        ep: Number of test episodes to be considered
        model_file: File for restoring model which is to be evaluated

        Output:
        Mean and standard deviation of rewards for the test episodes
        '''
        if model_file!=None:
            self.network.load_model(model_file)
        reward_list = np.zeros((ep,1))
        for episode in range(ep):
            done = False
            state = self.env.reset()
            while not done:
                q_values = self.network.model.predict(np.reshape(state,[1,self.env.observation_space.shape[0]]))
                action = self.epsilon_greedy_policy(q_values,0.05)
                next_state, reward, done, info =self.env.step(action)
                state = next_state
                reward_list[episode,0] = reward_list[episode,0] + reward
        return np.mean(reward_list), np.std(reward_list)
    
    def burn_in_memory(self):
        '''
        Function to initialize replay memory buffer with transition tuples generated by randomly initialized agent

        Parameters:
        None
        '''
        burn_in=10000
        state = self.env.reset()
        done = False
        iterations=0
#        for i in range(burn_in):
        while iterations < burn_in:
            if not done:
                
                action = self.env.action_space.sample()
                next_state, reward, done, info =self.env.step(action)
                self.memory.append((state,action,reward,next_state,done))
                state = next_state
                iterations=iterations+1
            else:
                state = self.env.reset()
                done = False
        

def parse_arguments():
    '''
    Function to parse arguments as per argparse

    Parameters:
    None

    Output:
    Parsed arguments
    '''
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()

def main(args):
    '''
    Function to initiate the training process and evaluating performance

    Parameters:
    args: arguments specifying environment, rendering, model file
    '''
    args = parse_arguments()
    environment_name = args.env
    render_decision = args.render
    train_decision = args.train
    model = args.model_file

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)
    
    # Creating instance of agent
    agent = DQN_Agent(environment_name,render_decision)
    if train_decision:
        agent.burn_in_memory()
        performance = agent.train()

    plt.plot(performance)
    name = 'double_DQN_plot_'+environment_name+'.png'
    plt.savefig(name)
    plt.show()
    reward,std = agent.test(100)
    print("The average reward is {} and std is {}".format(reward,std))

    
    if model!=None:    
        test_run = agent.test(100,model)
        print(test_run)
    

if __name__ == '__main__':
    main(sys.argv)
    

