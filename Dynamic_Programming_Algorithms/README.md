# Dynamic Programming based algorithms
This repository contains implementation for learning deterministic policies using
- Value iteration
- Policy iteration
## Background

## Usage
Following command can be used to train agent using policy iteration with default parameters in FrozenLake environment
```
python value_policy_iteration.py
```
Command line arguments can be used to specify parameters (or value iteration algorithm) which can be explored with `--h`

## Results and discussion
This implementation involved learning deterministic policies for FrozenLake (gym) environment. In deterministic mode of environment, i.e. without any frozen blocks, the agent was able to solve the environment. Following is the policy learnt in 4x4 deterministic frozen lake environment.  
{add image}  
However, deterministic policies are unable to solve stochastic FrozenLake consistently.
