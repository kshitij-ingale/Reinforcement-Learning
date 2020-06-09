""" Script for test functions using pytest """
import gym
import pytest
import numpy as np
from td_agent import TDL, parse_arguments


@pytest.fixture
def tdagent():
    args = parse_arguments().parse_args(["--e", "NChain-v0"])
    env = gym.make(args.environment_name)
    mcagent = TDL(env, args)
    yield mcagent
    env.close()


def test_QTable_shape(tdagent):
    """ Test if Q values array is intialized with correct shape

    Parameters
    ----------
    tdagent : instance of TDL class
        Agent instance intialized through fixture
    """
    assert tdagent.Q_values.shape == (5, 2)

