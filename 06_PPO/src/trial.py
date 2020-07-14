""" Script for test functions using pytest """
import gym
import pytest
import numpy as np
import tensorflow as tf
from network import Net
from ppo_agent import PPO, parse_arguments, TrainingParameters


args = parse_arguments().parse_args(["--e", "CartPole-v0", "--ep", "1"])
env = gym.make(args.environment_name)
agent = PPO(env, args)
state = env.reset()

# def env_step_fn(a):
#     return a, 2, done, {}
# agent.env.step = env_step_fn
# agent.policy_net.get_action = get_action_fn
# agent.state_function_estimator = estimate_vf_fn
# for _ in range(4):
#     states, actions, value_function_targets, done = agent.simulate_agent_for_max_timesteps(state)
#     print(value_function_targets)
#     if done:
#         state = env.reset()
#     else:
#         state = states[-1]
# env.close()

# vf = 5
# [2, 2, 2,
# 2, 4]
# expected:
# [(2 + 2*0.5 + 2*0.25 + 5*0.0625), (2 + 2*0.5 + 5*0.25), (2*0.25 + 5*0.5)]
# [(2 + 4*0.5), 4]

# class FakeEnv:
#     def __init__(self):
#         self.reward_sequence = [2, 2, 2, 2, 4]
#         self.ptr = 0
    
#     def step(self, idx):
#         print(self.ptr)
#         reward = self.reward_sequence[self.ptr]
#         self.ptr += 1
#         done=False
#         if self.ptr == len(self.reward_sequence):
#             done = True
#         return state, reward, done, None
def fake_env_step(idx):
    reward_sequence = [2, 2, 2, 2, 4]
    return idx + 1, reward_sequence[idx], idx == len(reward_sequence) - 1, None

def get_action_fn(idx,deterministic_policy):
    return int(tf.squeeze(idx).numpy())

def estimate_vf_fn(s):
    return 5

TrainingParameters.max_timesteps = 3

agent.discount = 0.5
# agent.env = FakeEnv()
agent.env.step = fake_env_step
agent.policy_net.get_action = get_action_fn
agent.state_function_estimator = estimate_vf_fn
state = 0
for _ in range(2):
    states, actions, value_function_targets, done, next_state = agent.simulate_agent_for_max_timesteps(state)
    print('states',states)
    print('vf',value_function_targets, '\n')
    if done:
        state = 0
    else:
        state = next_state
print([(2 + 2*0.5 + 2*0.25 + 5*0.125), (2 + 2*0.5 + 5*0.25), (2 + 5*0.5)])
print([(2 + 4*0.5), 4])

env.close()