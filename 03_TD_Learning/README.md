# TD Learning

## Background
SARSA and Q-learning

## Structure

```
.
├── arrays      # Store policy and Q-values
├── output      # Timestamped folders for runs containing logs and output artifacts
└── src
    └── *.py    # Source code files
```
## Usage
Train Monte Carlo agent using the following
```
python src/td_agent.py
```
Additional parameters can be configure through CLI arguments and `config.py`  
Run test functions with `pytest` for the implementation
```
pytest src/test.py
```

## Results and discussion
Taxi environment 