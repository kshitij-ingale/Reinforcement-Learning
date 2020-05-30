""" Implementation of TD-learning (SARSA and Q-learning)"""
import argparse, gym, os, logging, time, pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
from utils.utils import parse_config


class TDL:
    def __init__(self, env, parameters, logger=None):
        """ TD-learning agent

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
        num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n

        if parameters.saved_arrays_path:
            with open(parameters.saved_arrays_path, "rb") as f:
                saved_arrays = pickle.load(f)
                self.Q_values = saved_arrays["Qvalues"]
                self.state_idx_map = saved_arrays["statemap"]
        else:
            self.Q_values = np.zeros((num_states, self.num_actions))
            self.state_idx_map = {}
        self.state_idx_pointer = len(self.state_idx_map)
        if logger:
            self.print_fn = logger.info
        else:
            self.print_fn = print
        self.test_only = parameters.test_decision
        self.train_episodes = parameters.num_episodes

    def get_state_idx(self, state):
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
        if state not in self.state_idx_map:
            self.state_idx_map[state] = self.state_idx_pointer
            self.state_idx_pointer += 1
        return self.state_idx_map[state]

    def simulate_episodes(self):
        """ Simulating episodes and training agent in environment
        This function runs training for TD-learning and saves arrays of learnt Q-values
        and policy
        """

        algorithm = "Q_learning" if TrainingParameters.use_Q_learning else "SARSA"
        self.print_fn(
            f"Simulating {self.train_episodes} episodes to get Q-values with {algorithm}"
        )
        epsilon = TrainingParameters.initial_epsilon
        # anneal = (epsilon - TrainingParameters.final_epsilon) / TrainingParameters.decay_steps
        training_rewards = []
        for episode in tqdm(range(self.train_episodes)):
            done = False
            state = self.env.reset()
            while not done:
                state_idx = self.get_state_idx(state)
                action = self.epsilon_greedy_policy(
                    self.Q_values[state_idx, :], epsilon
                )
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self.get_state_idx(next_state)
                update_value = reward - self.Q_values[state_idx, action]
                if not done:
                    if TrainingParameters.use_Q_learning:
                        update_value += TrainingParameters.discount * max(
                            self.Q_values[next_state_idx, :]
                        )
                    else:
                        next_action = self.epsilon_greedy_policy(
                            self.Q_values[next_state_idx, :], epsilon
                        )
                        update_value += (
                            TrainingParameters.discount
                            * self.Q_values[next_state_idx, next_action]
                        )
                self.Q_values[state_idx, action] += (
                    float(TrainingParameters.learning_rate) * update_value
                )
                state = next_state
            if (episode % TrainingParameters.test_frequency == 0) and (episode > 0):
                reward, _ = self.execute_policy(
                    current_training_episode=episode,
                    test_episodes=TrainingParameters.test_episodes,
                    render=False,
                    save_frequency=TrainingParameters.video_save_frequency,
                )
                training_rewards.append(reward)

            # Anneal exploration epsilon by linear or exponential decay
            # epsilon = max(TrainingParameters.final_epsilon, epsilon - anneal)
            epsilon = max(
                TrainingParameters.final_epsilon,
                epsilon * TrainingParameters.decay_rate,
            )

        base_fname = f"{self.env.spec._env_name}_{algorithm}"

        # Save avg rewards obtained using greedy policy during training
        plt.figure(figsize=(15, 10))
        plt.plot(
            [
                ep
                for ep in range(1, self.train_episodes)
                if not (ep % TrainingParameters.test_frequency)
            ],
            training_rewards,
        )
        plt.xlabel("Number of episodes trained")
        plt.ylabel("Average reward obtained using greedy policy")
        plt.title(f"Rewards obtained during training with {algorithm}")
        plt.savefig(
            os.path.join(Directories.output, f"{base_fname}_training_rewards.png")
        )

        # Save Q-values calculated
        arrays_save_file = os.path.join(Directories.arrays, f"{base_fname}_arrays.pkl")
        self.print_fn(f"Saving Q-values in {arrays_save_file}")
        data_to_be_saved = {"Qvalues": self.Q_values, "statemap": self.state_idx_map}
        with open(arrays_save_file, "wb") as f:
            pickle.dump(data_to_be_saved, f)

    @staticmethod
    def greedy_policy(q_values):
        """ Returns action to be executed by agent as per greedy policy over Q-values

        Parameters
        ----------
        q_values : numpy.ndarray
            Q values for all actions in the current state of agent

        Returns
        -------
        int
            Action as per greedy policy in the current state of agent
        """
        return np.argmax(q_values)

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        """ Returns action to be executed by agent as per epsilon-greedy policy

        Parameters
        ----------
        q_values : numpy.ndarray
            Q values for all actions in the current state of agent
        epsilon : float
            Probability of random action (for exploration)

        Returns
        -------
        int
            Action as per epsilon-greedy policy in the current state of agent
        """
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(q_values)

    def execute_policy(
        self,
        current_training_episode=None,
        test_episodes=100,
        render=False,
        save_frequency=None,
    ):
        """ Run inference in environment using current policy and evaluate agent performance

        Parameters
        ----------
        current_training_episode : int, optional
            Current training episode number for saving progress video, by default None
        test_episodes : int, optional
            Number of test episodes to be simulated, by default 100
        render : bool, optional
            Render environment to stdout (or output window if render is implemented for environment), by default False
        save_frequency : int, optional
            specifies episode number out of test_episodes for which video is to be saved, if None, dont save any video,
            by default None

        Returns
        -------
        float
            Mean and std of rewards obtained by agent over number of test episodes specified
        """
        if save_frequency:

            def video_frequency(x):
                return x % save_frequency == 0

            if self.test_only:
                video_save_path = os.path.join(Directories.output, "videos")
            else:
                video_save_path = os.path.join(
                    Directories.output,
                    f"Training_progress_videos/{current_training_episode}",
                )
            self.env = gym.wrappers.Monitor(
                self.env, video_save_path, video_callable=video_frequency, force=True
            )
        rewards = np.zeros(test_episodes)
        for test_episode in range(test_episodes):
            curr_episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                if state not in self.state_idx_map:
                    # Use random action for unknown state
                    self.print_fn("Unknown state encountered, using random action")
                    action = self.env.action_space.sample()
                else:
                    action = self.greedy_policy(
                        self.Q_values[self.get_state_idx(state), :]
                    )
                state, reward, done, _ = self.env.step(action)

                curr_episode_reward += reward
            rewards[test_episode] = curr_episode_reward
        return np.mean(rewards), np.std(rewards)


def parse_arguments():
    """ Parse command line arguments using argparse """
    parser = argparse.ArgumentParser(description="Train agent using TD-learning")
    parser.add_argument(
        "--t", dest="test_decision", action="store_true", help="Test the agent"
    )
    parser.add_argument(
        "--e",
        dest="environment_name",
        default="Taxi-v3",
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
        dest="num_episodes",
        default=None,
        type=int,
        help="Number of episodes to be simulated for training (testing if test argument provided)",
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


def run_agent(args, logger):
    """ Run TD learning agent (train / inference)

    Parameters
    ----------
    args : argparse Namespace
        CLI arguments provided with script with argparse
    logger : logging.Logger
        logging object to be used for printing logs to stdout and writing to file
    """
    with gym.make(args.environment_name) as env:
        agent = TDL(env, args, logger)
        if not args.test_decision:
            agent.simulate_episodes()
            mean_reward, std_reward = agent.execute_policy(render=args.render_decision)
        else:
            mean_reward, std_reward = agent.execute_policy(
                args.num_episodes,
                render=args.render_decision,
                save_frequency=InferenceParameters.video_save_frequency,
            )
    agent.print_fn(
        f"The reward is {mean_reward} with standard deviation of {std_reward}"
    )


config_parameters = parse_config("config/config.yml")
TrainingParameters = config_parameters.Training
InferenceParameters = config_parameters.Inference
Directories = config_parameters.Directories

if __name__ == "__main__":

    TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args = parse_arguments().parse_args()
    if args.test_decision:
        if args.num_episodes is None:  # Set default test episodes to 100
            args.num_episodes = 100
        if args.saved_arrays_path is None:
            raise FileNotFoundError("Model path missing for inference")
        # Use saved model path directory
        Directories.output = "/".join(args.saved_arrays_path.split("/")[:-2])
        logger_file_path = os.path.join(Directories.output, "test.log")
    else:
        if args.num_episodes is None:  # Set default train episodes to 10000
            args.num_episodes = 10000
        if args.saved_arrays_path:  # Continue training from pretrained model
            Directories.output = "/".join(args.saved_arrays_path.split("/")[:-2])
        else:
            Directories.output = os.path.join(Directories.output, TIMESTAMP)
            os.makedirs(Directories.output, exist_ok=True)
        logger_file_path = os.path.join(Directories.output, "train.log")
    logger = setup_logging(logger_file_path)
    run_agent(args, logger)
