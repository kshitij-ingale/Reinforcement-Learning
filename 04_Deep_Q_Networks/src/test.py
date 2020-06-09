""" Script for test functions using pytest """
import gym
import pytest
import numpy as np
import tensorflow as tf
from network import Net
from dqn_agent import DQN, parse_arguments


def test_network_generation():
    """ Test if generation of neural networks is correct
    """

    class trial_net_params:
        hidden_units = [32, 32]
        normalization = None
        activation = "leaky_relu"
        learning_rate = 1e-2

    net_params = trial_net_params()
    trialNet = Net(4, 2, net_params, "trial")
    random_input = np.random.rand(1, 4)
    output = trialNet(random_input)
    assert output.shape == (1, 2)
    tf.debugging.assert_type(output, tf.float32)
    
# @pytest.fixture
# def dqn():
#     args = parse_arguments().parse_args(["--e", "CartPole-v0", "--ep", "1"])
#     env = gym.make(args.environment_name)
#     dqn = DQN(env, args)
#     yield dqn
#     env.close()
