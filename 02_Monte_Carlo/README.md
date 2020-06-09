# Monte Carlo algorithm
This implementation uses first visit policy evaluation based on-policy Monte Carlo algorithm.
## Background
Monte Carlo algorithms approximate value functions of actions or states by relying on simulating agent in environment and obtaining expected rewards from each state encountered in the process. Hence, these methods unlike dynamic programming, do not require a complete model of environment. The expected reward for each state observed in an episode can be approximated by either computing averages of discounted rewards obtained at first visit or every visit. These methods also follow an iterative approach of policy evaluation followed by policy improvement.  
Another aspect of Monte Carlo methods is exploring starts and according to that, all states (or state-action pairs) should have some probabilty of being the starting point of the episodes. However, this can be inefficient as episodes starting with certain states (or state-action pairs) can be illegal as per environment. A work-around for that is to learn epsilon soft greedy policy while biasing learning in favour of greedy policy.  
Off policy algorithms involve learning a target policy while exploring with behavior policy. This enables agent to avoid learning a sub-optimal policy (sub-optimal because of the exploration part) as well as leverage past experience. In order to train agent in off-policy mode using Monte Carlo methods, importance sampling is used. In simple terms, it weighs returns in ratio of probability of action in target and behavior policy.  
The value functions computed by these methods are unbiased estimates of the true value function. However, Monte Carlo methods suffer from high variance which decreases with higher number of simulations. In contrast, dynamic programming based methods have lower variance but high bias.

## Structure

```
.
├── Blackjack.ipynb   # Blackjack policy analysis notebook
├── config            # Folder to store config file
├── output            # Timestamped folders for runs containing learnt policy arrays, logs and output artifacts
└── src
    └── *.py          # Source code files
```
## Usage
Train Monte Carlo agent using the following
```
python src/mc_agent.py
```
Environment related arguments like environment name and rendering can be specifed through command line, other training related parameters and directories can be specified through `config.yml` file
Run test functions with `pytest` for the implementation
```
pytest src/test.py
```

## Results and discussion
For this implementation, blackjack environment was used to analyse performance. The jupyter notebook explores the value functions as well as the policy learnt by the agent 
