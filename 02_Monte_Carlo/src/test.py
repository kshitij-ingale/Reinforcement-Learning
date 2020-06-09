""" Script for test functions using pytest """
import gym
import pytest
import numpy as np
from mc_agent import MonteCarlo, parse_arguments


@pytest.fixture
def mcagent():
    args = parse_arguments().parse_args(["--e", "NChain-v0"])
    env = gym.make(args.environment_name)
    mcagent = MonteCarlo(env, args)
    yield mcagent
    env.close()


def test_QTable_shape(mcagent):
    """ Test if Q values array is intialized with correct shape

    Parameters
    ----------
    mcagent : instance of MonteCarlo class
        Agent instance intialized through fixture
    """
    assert mcagent.Q_store.Q_values.shape == (5, 2)


def test_discounted_returns(mcagent):
    """ Test if discounted rewards calculation is correct

    Parameters
    ----------
    mcagent : instance of MonteCarlo class
        Agent instance intialized through fixture
    """
    rewards_sequence = [2, 4, -6, 8]
    undiscounted_returns = [8.0, 6.0, 2.0, 8.0]
    discount_factor = 0.5
    discounted_returns = [3.5, 3.0, -2.0, 8.0]
    np.testing.assert_array_equal(
        mcagent.get_discounted_returns(rewards_sequence, 1.0),
        np.asarray(undiscounted_returns),
    )
    np.testing.assert_array_equal(
        mcagent.get_discounted_returns(rewards_sequence, discount_factor),
        np.asarray(discounted_returns),
    )
