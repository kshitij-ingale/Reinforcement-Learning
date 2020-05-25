# Monte Carlo algorithm

## Background
This algorithm uses first visit policy evaluation  
Exploring starts for MC, workaround is to use epsilon greedy policy

## Structure

```
.
├── arrays      # Store policy and Q-values
├── notebooks   # Notebooks for inference and analysis
├── output      # Timestamped folders for runs containing logs and output artifacts
└── src
    └── *.py    # Source code files
```
## Usage
Train Monte Carlo agent using the following
```
python src/mc_agent.py
```
Additional parameters can be configure through CLI arguments and `config.py`  
Run test functions with `pytest` for the implementation
```
pytest src/test.py
```

## Results and discussion
Blackjack policy is analysed in the notebook
