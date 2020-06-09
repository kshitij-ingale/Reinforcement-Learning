# TD Learning
This implementation focuses on temporal difference learning methods like Q-learning and SARSA in tabular setting.

## Background
Temporal Difference learning (TD-learning) is like a hybrid of planning and simulation based methods. These methods involve simulating agent for some steps and bootstrapping value function from the resulting state. A model of environment is not required for learning (as in case of dynamic programming) and the agent doesn't have to wait till the end of episode to update value functions as well (like in case of Monte Carlo methods)  
An example of TD-learning algorithm is SARSA. In this method, the agent starts from the current state S<sub>t</sub>, executes an action A<sub>t</sub> as per the policy and obtains reward R<sub>t</sub> while reaching the next state S<sub>t+1</sub>. In order to update the Q-value function, the agent then uses the Q-value function corresponding to next state (and action A<sub>t+1</sub> from the next state as per the policy). Since the agent behaves as per the same policy as it is using to learn, it is learning in an on-policy way.  
Another example of TD-learning is Q-learning. This is like an off-policy variant which uses the best Q-value (greedy policy) from next state to update the current state Q-value (instead of Q-value corresponding to A<sub>t+1</sub> as per behaviour or non-greedy policy)


## Structure

```
.
├── config      # Folder to store config file
├── output      # Timestamped folders for runs containing logs and output artifacts
└── src
    └── *.py    # Source code files
```
## Usage
Train TD-learning agent using the following
```
python src/td_agent.py
```
Environment related arguments like environment name and rendering can be specifed through command line, other training related parameters and directories can be specified through `config.yml` file. The config file allows to specify whether SARSA or Q-learning is to be used.
  
Run test functions with `pytest` for the implementation
```
pytest src/test.py
```

## Results and discussion
This implementation used the taxi environment to evaluate performance. The task to pick-up passenger from a destination (indicated by blue) and drop them off at another location (indicated by magenta). The agent gets a reward of +20 for dropping passenger off at correct location and -1 for any action with exception of illegal drop-off and pick-up yielding reward of -10. As it can be seen in the following gif that the agent learns the policy to complete this task

{add gif}