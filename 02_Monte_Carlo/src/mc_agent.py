""" Script for training agent with Monte Carlo algorithm """
import gym
import argparse, os, logging, time, pickle
import numpy as np
from tqdm import tqdm
import sys

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
from utils.utils import parse_config


class QTable:
    def __init__(self, num_states, num_actions):
        """ Arrays for storing policy, number of visits, Q-values
        Since state obtained from gym environment can be of any type, a dictionary
        mapping state to an index in the Q-values, policy and visits arrays is used

        Parameters
        ----------
        num_states : int
            Number of possible states in environment
        num_actions : int
            Number of actions for agent in environment
        """
        # self.Q_values = np.random.rand(num_states, num_actions)
        self.Q_values = np.zeros((num_states, num_actions))
        self.state_map = {}
        self.pointer = 0
        self.policy = np.ones((num_states, num_actions)) * (1 / num_actions)
        self.visits = np.zeros((num_states, num_actions))

    def get_index(self, state):
        """ Returns index corresponding to state in arrays
        In order to map an index in array to state, dictionary of states mapping to index in array 
        is maintained. If a new state is encountered, an index is allocated to that state.

        Parameters
        ----------
        state : any
            State for which index is to be obtained, it can be of any type as provided by environment

        Returns
        -------
        int
            Index for (Q-values, policy, visits) arrays corresponding to state
        """
        if state not in self.state_map:
            self.state_map[state] = self.pointer
            self.pointer += 1
        return self.state_map[state]

    def save_current_arrays(self, filepath):
        """ Saves Q_values, policy, visits and mapping dictionary of states to index in these
        arrays to a pkl file

        Parameters
        ----------
        filepath : str
            File name (with path) to pkl file for saving arrays
        """
        arrays = {}
        arrays["qvalues"] = self.Q_values
        arrays["statemap"] = self.state_map
        arrays["policy"] = self.policy
        arrays["visits"] = self.visits
        with open(filepath, "wb") as f:
            pickle.dump(arrays, f)

    def load_arrays(self, saved_arrays_path):
        """ Loads Q_values, policy, visits and mapping dictionary of states to index in these
        arrays from a pkl file

        Parameters
        ----------
        saved_arrays_path : str
            File name (with path) to pkl file for loading arrays
        """
        with open(saved_arrays_path, "rb") as f:
            arrays = pickle.load(f)
        self.Q_values = arrays["qvalues"]
        self.state_map = arrays["statemap"]
        self.policy = arrays["policy"]
        self.visits = arrays["visits"]


class MonteCarlo:
    def __init__(self, env, parameters, logger=None):
        """ Monte Carlo agent

        Parameters
        ----------
        env : gym environment instance
            Gym environment to be used for Monte Carlo agent
        parameters : arparse arguments
            CLI arguments provided to script parsed through argparse
        logger : logging.Logger object, optional
            Logging object to be used, if not provided, print to stdout, by default None
        """
        self.env = env
        if (not isinstance(self.env.action_space, gym.spaces.discrete.Discrete)) or (
            not isinstance(self.env.action_space, gym.spaces.discrete.Discrete)
        ):
            raise NotImplementedError(
                "Continuous observation or action space is not supported"
            )
        num_states = 1
        if hasattr(self.env.observation_space, "spaces"):
            for space in self.env.observation_space.spaces:
                num_states *= space.n
        else:
            num_states *= self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.Q_store = QTable(num_states, self.num_actions)

        if parameters.saved_arrays_path:
            self.Q_store.load_arrays(parameters.saved_arrays_path)
        if logger:
            self.print_fn = logger.info
        else:
            self.print_fn = print
        self.train_episodes = parameters.train_episodes

    def simulate_episodes(self):
        """ Monte Carlo simulations of episodes
        This function runs training for MC and saves arrays of learnt Q-values and policy
        """
        self.print_fn(f"Simulating {self.train_episodes} episodes to get Q-values")
        for _ in tqdm(range(self.train_episodes)):
            done = False
            state = self.env.reset()
            state_actions, rewards = [], []
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state_actions.append((state, action))
                rewards.append(reward)
                state = next_state
            self.update_q_values_and_policy(state_actions, rewards)
        arrays_save_file = os.path.join(
            Directories.output, f"{self.env.spec._env_name}_arrays.pkl"
        )
        self.print_fn(f"Saving Q-values and policy arrays in {arrays_save_file}")
        self.Q_store.save_current_arrays(arrays_save_file)

    @staticmethod
    def get_discounted_returns(rewards, discount):
        """ Returns discounted future returns for each state in the provided rewards
        sequence

        Parameters
        ----------
        rewards : list
            Sequence of rewards achieved by agent in an episode
        discount : float
            Discount factor (gamma) to be used for discounting future rewards

        Returns
        -------
        numpy.ndarray
            Discounted future returns for each state in rewards sequence
        """
        discounted_rewards = np.zeros(len(rewards))
        discounted_rewards[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            discounted_rewards[i] = discount * discounted_rewards[i + 1] + rewards[i]
        return discounted_rewards

    def update_q_values_and_policy(self, state_actions, rewards):
        """ Updates Q-values and policy arrays after simulating an episode as per first visit policy evaluation

        Parameters
        ----------
        state_actions : list
            List of (state, action) tuples in an episode
        rewards : list
            Sequence of rewards achieved by agent in an episode
        """
        discounted_returns = self.get_discounted_returns(
            rewards, TrainingParameters.discount
        )
        first_occurrence = set()
        for idx, (state, action) in enumerate(state_actions):
            if (state, action) not in first_occurrence:
                first_occurrence.add((state, action))
                state_index = self.Q_store.get_index(state)
                # Incremental update for mean of Q-values
                self.Q_store.visits[state_index, action] += 1
                self.Q_store.Q_values[state_index, action] += (
                    1 / self.Q_store.visits[state_index, action]
                ) * (
                    discounted_returns[idx] - self.Q_store.Q_values[state_index, action]
                )

                # Update policy with epsilon probability for random action and additional (1 - epsilon) for greedy action
                self.Q_store.policy[state_index, :] = (
                    TrainingParameters.epsilon / self.num_actions
                )
                self.Q_store.policy[
                    state_index, np.argmax(self.Q_store.Q_values[state_index, :])
                ] += (1 - TrainingParameters.epsilon)

    def get_action(self, state):
        """ Samples action as per action probabilities for the state in policy

        Parameters
        ----------
        state : any
            State for which index is to be obtained, it can be of any type as provided by environment

        Returns
        -------
        int
            Discrete action sampled as per policy
        """
        return np.random.choice(
            np.arange(self.num_actions),
            p=self.Q_store.policy[self.Q_store.get_index(state), :],
        )

    def execute_policy(self, test_episodes=100, render=False):
        """ Run inference in environment using current policy

        Parameters
        ----------
        test_episodes : int, optional
            Number of test episodes to be simulated, by default 100
        render : bool, optional
            Render environment to stdout (or output window if render is implemented for environment), by default False

        Returns
        -------
        float
            Mean and std of rewards obtained by agent over number of test episodes specified
        """
        # Blackjack environment doesnt have default render, so additional print statements are used
        use_black_jack = (self.env.spec._env_name == "Blackjack") if render else False

        rewards = np.zeros(test_episodes)
        for test_episode in range(test_episodes):
            curr_episode_reward = 0
            state = self.env.reset()
            if use_black_jack:
                print(
                    f"Dealer's visible card:{state[1]} [with sum: {sum(self.env.dealer)}] "
                    f"and current sum of player: {state[0]} {'with' if state[2] else 'without'} usable ace"
                )
            done = False
            while not done:
                if render and not use_black_jack:
                    self.env.render()
                if state not in self.Q_store.state_map:
                    # Use random action for unknown state
                    self.print_fn("Unknown state encountered, using random action")
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(
                        self.Q_store.Q_values[self.Q_store.get_index(state), :]
                    )
                    # action = self.get_action(state) # Sample as per action probabilities of policy
                state, reward, done, _ = self.env.step(action)
                if use_black_jack:
                    print(
                        f"Agent decides to {'hit' if action else 'stick'} and gets a reward of {reward}\n"
                        f"Now, [dealer's sum: {sum(self.env.dealer)}] and current sum of player: {state[0]}"
                        f" {'with' if state[2] else 'without'} usable ace"
                    )
                curr_episode_reward += reward
            rewards[test_episode] = curr_episode_reward
            if use_black_jack:
                print(f"Episode finished with total reward of {curr_episode_reward}\n")
        self.print_fn(
            f"Mean reward {np.mean(rewards)} with std of {np.std(rewards)} obtained for {test_episodes} test episodes"
        )


def parse_arguments():
    """ Parse command line arguments using argparse """
    parser = argparse.ArgumentParser(description="Monte Carlo Agent")
    parser.add_argument(
        "--t", dest="test_decision", action="store_true", help="Test the agent"
    )
    parser.add_argument(
        "--e",
        dest="environment_name",
        default="Blackjack-v0",
        type=str,
        help="Gym Environment",
    )
    parser.add_argument(
        "--m",
        dest="saved_arrays_path",
        type=str,
        help="Path to saved Q-values, policy arrays",
    )
    parser.add_argument(
        "--r",
        dest="render_decision",
        action="store_true",
        help="Visualize environment while simulating policy",
    )
    parser.add_argument(
        "--ep",
        dest="train_episodes",
        default=10000,
        type=int,
        help="Number of Training episodes",
    )
    return parser


def setup_logging(logger_file_path):
    """ Returns logging object which writes logs to a file and stdout

    Parameters
    ----------
    logger_file_path : str
        Path (with filename) to the log file

    Returns
    -------
    logging object
        Configured logging object for logging
    """
    # Create logging object
    logger = logging.getLogger(name=__name__)
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


config_parameters = parse_config("config/config.yml")
TrainingParameters = config_parameters.Training
Directories = config_parameters.Directories


if __name__ == "__main__":

    TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # Create a timestamped output directory to store artifacts
    Directories.output = os.path.join(Directories.output, TIMESTAMP)
    os.makedirs(Directories.output)

    logger = setup_logging(os.path.join(Directories.output, "training.log"))
    args = parse_arguments().parse_args()

    env = gym.make(args.environment_name)
    mcagent = MonteCarlo(env, args, logger)
    if not args.test_decision:
        mcagent.simulate_episodes()
    mcagent.execute_policy(render=args.render_decision)
