#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, random
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here. 
        
        env = gym.make(environment_name)
        
        if environment_name == 'CartPole-v0':
            learning_rate = 0.001
            neurons_H1 = 16
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
            neurons_H2 = 16
            input_layer = Input(shape=env.observation_space.shape)
            hidden_layer1 = Dense(neurons_H1,activation='relu',kernel_initializer = 'he_uniform')(input_layer)
            hidden_layer2 = Dense(neurons_H2,activation='relu',kernel_initializer = 'he_uniform')(hidden_layer1)
            output_layer = Dense(env.action_space.n,activation='linear')(hidden_layer2)

        model = Model(inputs=input_layer, outputs=output_layer)

        adam = optimizers.Adam(lr = learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        
        self.model = model
        
    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(suffix)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = load_model(model_file)

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(weight_file)

class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum sjze after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 

       # self.memory = np.zeros([memory_size,4])

       self.memory = []
       self.current = 0
       self.max = memory_size
#       self.burn = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        
        return random.sample(self.memory,batch_size)

    def append(self, transition):
        # Appends transition to the memory.
        if self.current<self.max:
            # self.memory[self.current,:] = transition
            self.memory.append(transition)
            self.current = self.current + 1
        else:
            self.memory[self.current%self.max] = transition
            self.current = self.current + 1

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, render=False):
        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        
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
        # Creating epsilon greedy probabilities to sample from.         
        num = random.random()
        if(num<epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        return np.argmax(q_values)

    def train(self):
        
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        step = 0
        performance_reward = []
        if self.render_decision:
            self.env.render()
            def f(x):
                return x%3333==0
            self.env.render()
            path ='video_DQN_'+self.env_name+'/'
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
                    epsilon = 0.5-((step*(0.5-0.05))/5000000)
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
#               Find maximum q value
                q_max = np.amax(q,axis=1)
                
                for i in range(32):
                    if check[i]:
                        y[i,act[i]] = rew[i]
                    else:
                        y[i,act[i]] = rew[i] + self.discount*(q_max[i])
#                Store data in batches
                y_train = y
                x_train = np.array(c_stat)
#                Train network based on batches generated
                self.network.model.fit(x_train, y_train, epochs = 1,batch_size=32,verbose=0)
#                Find reward obtained in each episode
                total_reward = total_reward + reward
                step = step + 1

#           Evaluate performance after every 200 episodes
            if episode%200==0:
                mean,_ = self.test(20)
                performance_reward.append(mean)
#                Save model after every 1000 episodes
                if episode%1000==0 and self.env_name=='CartPole-v0':
                    name = 'DQN_model_'+self.env_name+'_'+str(episode)+'.h5'
                    self.network.model.save(name)
            if reward!=-200 and episode%100==0:
                print("After {} episodes, the reward in this episode is {}".format(episode,total_reward))
        return performance_reward
        

    def test(self, ep,model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
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
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        
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
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()

def main(args):
    

    args = parse_arguments()
    environment_name = args.env
    render_decision = args.render
    train_decision = args.train
    model = args.model_file

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)
    
    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    agent = DQN_Agent(environment_name,render_decision)
    if train_decision:
        agent.burn_in_memory()
        performance = agent.train()

    plt.plot(performance)
    name = 'DQN_plot_'+environment_name+'.png'
    plt.savefig(name)
    plt.show()
    reward,std = agent.test(100)
    print("The average reward is {} and std is {}".format(reward,std))

    
    if model!=None:    
        test_run = agent.test(100,model)
        print(test_run)
    

if __name__ == '__main__':
    main(sys.argv)
    

