""" Script for test functions using pytest """
import gym
import pytest
import numpy as np
import tensorflow as tf
from network import Net
from ppo_agent import PPO, parse_arguments, TrainingParameters


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
    

@pytest.fixture
def actor_critic():
    args = parse_arguments().parse_args(["--e", "CartPole-v0", "--ep", "1"])
    env = gym.make(args.environment_name)
    TrainingParameters.use_REINFORCE = False
    TrainingParameters.use_baseline = True
    actor_critic = PolicyGrad(env, args)
    yield actor_critic
    env.close()


def test_discounted_returns(reinforce):
    """ Test if discounted rewards calculation is correct

    Parameters
    ----------
    reinforce : instance of PolicyGrad class
        Agent instance for REINFORCE intialized through fixture
    """
    rewards_sequence = [2, 4, -6, 8]
    undiscounted_returns = [8.0, 6.0, 2.0, 8.0]
    discount_factor = 0.5
    discounted_returns = [3.5, 3.0, -2.0, 8.0]
    np.testing.assert_array_equal(
        reinforce.get_discounted_returns(rewards_sequence, 1.0),
        np.asarray(undiscounted_returns).reshape(-1, 1),
    )
    np.testing.assert_array_equal(
        reinforce.get_discounted_returns(rewards_sequence, discount_factor),
        np.asarray(discounted_returns).reshape(-1, 1),
    )


def test_episode_simulation_ac(actor_critic):
    """ Test if value function target corresponding to each state is obtained

    Parameters
    ----------
    actor_critic : instance of PolicyGrad class
        Agent instance for Actor-critic intialized through fixture
    """
    states, actions, value_function_targets = actor_critic.simulate_episode_for_nstep_returns()
    assert all(
        (
            len(states) == len(actions),
            len(states) == len(actions),
            len(states) == len(value_function_targets),
        )
    )

def test_outputs_for_Net(actor_critic):
    """ Test if action probabilities and value function estimate from Net are of correct shape and dtype

    Parameters
    ----------
    pgagent : instance of PolicyGrad class
        Agent instance for actor-critic intialized through fixture
    """
    state_tensor = tf.expand_dims(actor_critic.env.reset(), axis=0)
    action_probabilities = actor_critic.policy_net(state_tensor)
    assert action_probabilities.shape == (1, 2)
    tf.debugging.assert_type(action_probabilities, tf.float32)
    value_function = actor_critic.state_function_estimator(state_tensor)
    assert value_function.shape == (1, 1)
    tf.debugging.assert_type(value_function, tf.float32)

def test_inputs_for_reinforce_train_steps(reinforce_with_baseline):
    """ Test if input tensors to training steps in REINFORCE are of correct shape

    Parameters
    ----------
    reinforce_with_baseline : instance of PolicyGrad class
        Agent instance for REINFORCE with baseline intialized through fixture
    """
    states, actions, rewards = reinforce_with_baseline.simulate_episode_for_reinforce()
    timesteps_in_episode = len(states)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions)
    discounted_returns = tf.convert_to_tensor(
        reinforce_with_baseline.get_discounted_returns(
            rewards, reinforce_with_baseline.discount
        )
    )
    if hasattr(reinforce_with_baseline, "state_function_estimator"):
        estimated_value_function = reinforce_with_baseline.state_function_estimator(
            states
        )
        advantage_function = discounted_returns - estimated_value_function
        assert estimated_value_function.shape == (timesteps_in_episode, 1)
        assert advantage_function.shape == (timesteps_in_episode, 1)
    assert states.shape == (timesteps_in_episode, 4)
    assert actions.shape == (timesteps_in_episode,)
    assert discounted_returns.shape == (timesteps_in_episode, 1)


def test_inputs_for_ac_train_steps(actor_critic):
    """ Test if input tensors to training steps in actor critic are of correct shape

    Parameters
    ----------
    actor_critic : instance of PolicyGrad class
        Agent instance for actor-critic with baseline intialized through fixture
    """
    states, actions, value_function_targets = actor_critic.simulate_episode_for_nstep_returns()
    timesteps_in_episode = len(states)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions)
    if tf.is_tensor(value_function_targets[0]):
        value_function_targets = tf.concat(values=value_function_targets, axis=0)
    else:
        """
        If using MC returns (using high value of n in n-step return), convert floats to tensor and add axis in last dimension
        to convert shape of targets from (num_transitions,) to (num_transitions,1)
        """
        value_function_targets = tf.expand_dims(
            tf.convert_to_tensor(value_function_targets, dtype=tf.float32), axis=-1,
        )

    advantage_functions = (
        value_function_targets - actor_critic.state_function_estimator(states)
    )
    assert states.shape == (timesteps_in_episode, 4)
    assert actions.shape == (timesteps_in_episode,)
    assert value_function_targets.shape == (timesteps_in_episode, 1)
    assert advantage_functions.shape == (timesteps_in_episode, 1)
