""" Script for implementation of Policy and Value iteration algorithm """
import argparse, gym, logging, os
import matplotlib.pyplot as plt
import numpy as np
import sys
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
from utils.utils import parse_config

def evaluate_policy(
    ValueFunction, discount, policy=None, policy_iteration=True, asynchronous=False
):
    """ Perform policy evaluation step, in case of value iteration, value functions are updated based on 
    greedy action

    Parameters
    ----------
    ValueFunction : numpy.ndarray
        Value function for each state of environment
    discount : float
        Discount factor for future returns
    policy : numpy.ndarray, optional
        Action probabilities as per learnt policy in each state of environment, (since this implementation uses 
        deterministic policy, this would be boolean array indicating action to be executed in the state), 
        not required for value iteration and so by default None
    policy_iteration : bool, optional
        Train using policy iteration algorithm (else use value iteration), by default True
    asynchronous : bool, optional
        Update value functions in asynchronous mode (synchronous mode updates a copy of value function, so that original 
        value functions for all states are used for updating), by default False

    Returns
    -------
    numpy.ndarray
        Updated value function for each state of environment
    """
    if asynchronous:
        OriginalValueFunction = ValueFunction
    else:
        OriginalValueFunction = ValueFunction.copy()
    for state in range(env.observation_space.n):
        Q_values_state = []
        for action in range(env.action_space.n):
            TransitionTuplesList = env.P[state][action]
            Q_value_action = 0
            # Compute Q_value for action based on all possible next states
            for TransitionProb, NewState, Reward, done in TransitionTuplesList:
                Q_value_action += TransitionProb * (Reward)
                if not done:
                    Q_value_action += TransitionProb * (
                        discount * OriginalValueFunction[NewState]
                    )

            if policy_iteration:
                Q_values_state.append(policy[state, action] * Q_value_action)
            else:
                Q_values_state.append(Q_value_action)
        """
        For policy iteration, update value function by sum of Q-values while 
        for value iteration, update value function by considering max Q-value
        """
        if policy_iteration:
            ValueFunction[state] = sum(Q_values_state)
        else:
            ValueFunction[state] = max(Q_values_state)
    return ValueFunction


def improve_policy(ValueFunction, policy, discount):
    """ Perform policy improvement step by obtaining greedy actions over calculated
    value functions

    Parameters
    ----------
    ValueFunction : numpy.ndarray
        Value function for each state of environment
    policy : numpy.ndarray
        Action probabilities as per learnt policy in each state of environment, (since this implementation uses 
        deterministic policy, this would be boolean array indicating action to be executed in the state)
    discount : float
        Discount factor for future returns

    Returns
    -------
    numpy.ndarray, bool
        Updated policy, variable indicating if any changes were made to policy after improvement step
    """
    OriginalPolicy = policy.copy()
    for state in range(env.observation_space.n):
        StateQValues = []
        for action in range(env.action_space.n):
            TransitionTuplesList = env.P[state][action]
            Q_value_action = 0
            for TransitionProb, NewState, Reward, _ in TransitionTuplesList:
                Q_value_action += (
                    Reward + discount * TransitionProb * ValueFunction[NewState]
                )
            StateQValues.append(Q_value_action)
            policy[state, action] = 0
        policy[state, np.argmax(StateQValues)] = 1
    return policy, (policy == OriginalPolicy).all()


def execute_policy(policy, print_fn, render=False):
    """ Execute learnt policy in environment

    Parameters
    ----------
    policy : numpy.ndarray
        Action probabilities as per learnt policy in each state of environment, (since this implementation uses 
        deterministic policy, this would be boolean array indicating action to be executed in the state)
    print_fn : function, optional
        Function to be used for displaying reward obtained, if no logger provided, print to stdout
    render : bool, optional
        Render environment to stdout, by default False
    """
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        action = np.argmax(policy[state, :])
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print_fn(f"The policy yields a reward of {total_reward}")


def print_policy(policy, file_base_name, ValueFunction):
    """ Print greedy policy action overlayed on frozen lake environment and heat map of value functions
    of states in the environment

    Parameters
    ----------
    policy : numpy.ndarray
        Action probabilities as per learnt policy in each state of environment, (since this implementation uses 
        deterministic policy, this would be boolean array indicating action to be executed in the state)
    file_base_name : str
        File base name (with path) to saving image of policy actions overlayed on environment and value functions
        heat map
    ValueFunction : numpy.ndarray
        Value function for each state of environment
    """
    # Display grid consisting of Frozen Lake environment
    num_states_single_line = env.desc.shape[0]
    grid_data = np.zeros((num_states_single_line, num_states_single_line))
    for i in range(num_states_single_line):
        for j in range(num_states_single_line):
            if env.desc[i, j] == b"S":
                grid_data[i, j] = 1
            elif env.desc[i, j] == b"F":
                grid_data[i, j] = 0.2
            elif env.desc[i, j] == b"G":
                grid_data[i, j] = 0.7
    ax = plt.gca()
    plt.imshow(grid_data, cmap=plt.get_cmap("gnuplot"))
    # Overlay greedy actions in each state of Frozen Lake environment
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    for i in range(num_states_single_line):
        for j in range(num_states_single_line):
            greedy_action = np.argmax(policy[(i * num_states_single_line) + j, :])
            text = f'{action_names[greedy_action]}\n({env.desc[i, j].astype("<U1")})'
            ax.text(j, i, text, ha="center", va="center")
    plt.title("Policy learnt using DP")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{file_base_name}_policy.png")
    plt.figure()
    plt.imshow(ValueFunction.reshape((num_states_single_line, num_states_single_line)))
    plt.colorbar()
    plt.title("Value functions heatmap")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{file_base_name}_value_functions.png")


def run_DP_algorithm(parameters, logger=None):
    """ Train agent using specified dynamic programming algorithm
    Policy iteration involves multiple sweeps in state space while value iteration performs 
    single sweep to evaluate policy

    Parameters
    ----------
    parameters : argparse arguments
        Algorithm training parameters specified through CLI
    logger : logging.Logger object, optional
        Logging object to be used, if not provided, print to stdout, by default None

    Raises
    ------
    TimeoutError
        For value iteraton, if value function fails to converge (due to some issue), error is raised
    """
    if logger:
        print_fn = logger.info
    else:
        print_fn = print
    env_mode = "deterministic" if parameters.deterministic else "stochastic"
    algorithm = (
        "value_iteration" if parameters.run_value_iteration else "policy_iteration"
    )
    ValueFunction = np.zeros(env.observation_space.n)
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    print_fn(f"Training agent using {algorithm} in {env_mode} Frozen Lake environment")
    if parameters.run_value_iteration:
        MaxUpdate = Training_parameters.tolerance + 1
        num_iterations = 0
        # Calculate value function of states
        while MaxUpdate > Training_parameters.tolerance:
            UpdatedValueFunction = evaluate_policy(
                ValueFunction.copy(),
                Training_parameters.discount,
                policy_iteration=False,
                asynchronous=Training_parameters.asynchronous,
            )
            MaxUpdate = np.max(np.abs(ValueFunction - UpdatedValueFunction))
            ValueFunction = UpdatedValueFunction
            num_iterations += 1
            if num_iterations == Training_parameters.max_iterations:
                raise TimeoutError(
                    f"Value Function failed to converge after {Training_parameters.max_iterations} iterations"
                )
        print_fn(f"Value iteration required {num_iterations} steps to converge")
        # Obtain greedy policy based on calculated value functions
        policy, _ = improve_policy(ValueFunction, policy, Training_parameters.discount)
    else:
        # Initialize random policy
        np.put_along_axis(
            arr=policy,
            indices=np.random.randint(
                0, env.action_space.n, (env.observation_space.n, 1)
            ),
            values=1,
            axis=1,
        )
        PolicyConverged = False
        while not PolicyConverged:
            # Policy Evaluation step
            num_iterations = 0
            MaxUpdate = Training_parameters.tolerance + 1
            while (MaxUpdate > Training_parameters.tolerance) and (
                num_iterations < Training_parameters.max_iterations
            ):
                UpdatedValueFunction = evaluate_policy(
                    ValueFunction.copy(),
                    Training_parameters.discount,
                    policy,
                    asynchronous=Training_parameters.asynchronous,
                )
                MaxUpdate = np.max(np.abs(UpdatedValueFunction - ValueFunction))
                ValueFunction = UpdatedValueFunction
                num_iterations += 1
            # Policy Improvement step
            policy, PolicyConverged = improve_policy(
                ValueFunction, policy, Training_parameters.discount
            )
    # Run inference in environment using policy learnt by DP
    execute_policy(policy, print_fn, render=parameters.render_decision)
    file_base_name = f"{parameters.environment_name}_{env_mode}_{algorithm}"
    print_policy(
        policy, os.path.join(parameters.save_dir, file_base_name), ValueFunction
    )


def parse_arguments():
    """ Parse command line arguments using argparse """
    parser = argparse.ArgumentParser(description="Train Value/Policy iteration agent")
    parser.add_argument(
        "--e",
        dest="environment_name",
        default="FrozenLake-v0",
        type=str,
        help="Gym Environment to be used for training (FrozenLake-v0 or FrozenLake8x8-v0)",
    )
    parser.add_argument(
        "--d",
        dest="deterministic",
        action="store_true",
        help="Use deterministic environment (No frozen blocks in environment)",
    )
    parser.add_argument(
        "--r",
        dest="render_decision",
        action="store_true",
        help="Visualize environment while simulating policy",
    )
    parser.add_argument(
        "--vi",
        dest="run_value_iteration",
        action="store_true",
        help="Train using value iteration (policy iteration used if not set)",
    )
    parser.add_argument(
        "--dir",
        dest="save_dir",
        default="output",
        type=str,
        help="Directory to save artifacts",
    )
    return parser


def setup_logging(logger_file_path):
    """ Create a logging object storing logs in a file and displaying to stdout
    log is appended if file exits (pretraining)

    Parameters
    ----------
    logger_file_path : str
        Path (with filename) to the log file

    Returns
    -------
    logging.Logger
        Logging instance
    """
    # Create logging object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    # Save logs to file
    file_logging = logging.FileHandler(logger_file_path)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    file_logging.setFormatter(fmt)
    file_logging.setLevel(logging.DEBUG)
    logger.addHandler(file_logging)
    # Print log to stdout
    stdout_logging = logging.StreamHandler()
    stdout_logging.setFormatter(fmt)
    logger.addHandler(stdout_logging)
    return logger


if __name__ == "__main__":
    config_parameters = parse_config('config.yml')
    Training_parameters = config_parameters.Training
    args = parse_arguments().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.save_dir, "DP_algorithm.log"))
    with gym.make(args.environment_name, is_slippery=(not args.deterministic)) as env:
        run_DP_algorithm(args, logger)
