""" Script for defining training parameters and directories for Monte Carlo Agent """


class TrainingParameters:
    epsilon = 0.5       # Exploration probability for random action
    discount = 0.95     # Discount factor for future returns


class Directories:
    output = "output/"  # Directory to store rendered videos and logs
    arrays = "arrays/"  # Directory to store Q-value arrays for restoring or running MC inference
