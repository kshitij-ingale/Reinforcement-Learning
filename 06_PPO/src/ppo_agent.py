""" Script implementing Proximal Policy Optimization (PPO) agent """
import argparse, gym, os, time, logging, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from network import (
    Net,
    policy_net_train_step,
    state_function_estimator_train_step,
    calculate_likelihood,
)

np.random.seed(42)
from tqdm import tqdm
import sys

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
from utils.utils import parse_config


class PPO:
    def __init__(self, env, parameters, logger=None):
        """ Proximal Policy Optimization agent instance

        Parameters
        ----------
        env : gym environment instance
            Environment to be used for training or inference
        parameters : argparse.namespace
            CLI arguments provided as input through argparse
        logger : logging.Logger, optional
            logging instance to be used for writing logs to 
            file and stdout, by default None
        """
        # Gym environment parameters
        self.env_name = parameters.environment_name
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.entropy_weight = None
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.num_actions = self.env.action_space.n
            self.continuous_action_space = False
        else:
            self.continuous_action_space = True
            self.num_actions = self.env.action_space.shape[0]
            if TrainingParameters.entropy_weight:
                self.entropy_weight = tf.constant(
                    float(TrainingParameters.entropy_weight)
                )
        # Policy network
        self.policy_net = Net(
            self.state_dim,
            self.num_actions,
            NetworkParameters.policy,
            "actor",
            self.continuous_action_space,
        )
        self.render_decision = parameters.render_decision
        self.test_only = parameters.test_decision
        if logger is not None:
            self.logger = logger
            self.print_fn = self.logger.info
        else:
            self.print_fn = print
        if not self.test_only:
            # Training parameters
            self.train_iterations = parameters.num_episodes
            self.test_episodes = TrainingParameters.test_episodes
            self.test_frequency = TrainingParameters.test_frequency
            self.render_frequency = TrainingParameters.render_frequency
            self.video_save_frequency = TrainingParameters.video_save_frequency
            self.model_save_frequency = TrainingParameters.model_save_frequency
            self.discount = TrainingParameters.discount
            self.starting_iteration = 0
            self.state_function_estimator = Net(
                self.state_dim, 1, NetworkParameters.state_function_estimator, "critic",
            )
            self.pi_ratio = tf.constant(float(TrainingParameters.pi_ratio))
            self.print_fn("Initiating model for PPO algorithm")

        if parameters.model_path:
            self.policy_net.load_weights(parameters.model_path)
            self.print_fn(
                f"Loading weights for {self.policy_net.name} from {parameters.model_path}"
            )
            self.starting_iteration = int(parameters.model_path[-5:])
            if parameters.state_function_estimator_model_path:
                self.state_function_estimator.load_weights(
                    parameters.state_function_estimator_model_path
                )
                self.print_fn(
                    f"Loading weights for {self.state_function_estimator.name} "
                    f"from {parameters.state_function_estimator_model_path}"
                )

    def simulate_agent_for_max_timesteps(self, state):
        states, rewards, actions = [], [], []
        timestep, done = 0, False
        while not done and (timestep < TrainingParameters.max_timesteps):
            action = self.policy_net.get_action(
                tf.expand_dims(state, axis=0), deterministic_policy=False
            )
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            timestep += 1
        R = 0
        if not done:
            # Bootstrap value function for last state (if not terminal)
            R = self.state_function_estimator(tf.expand_dims(state, axis=0))
        value_function_targets = np.zeros((len(states), 1))
        for updatestep in range(timestep - 1, -1, -1):
            R = rewards[updatestep] + self.discount * R
            value_function_targets[updatestep] = R
        return states, actions, value_function_targets, done, next_state

    def train(self):
        """ Train agent using PPO algorithm
        """
        tensorboard_dir = os.path.join(Directories.output, "tensorboard")
        tensorboard_file_writer = tf.summary.create_file_writer(tensorboard_dir)
        # Actor optimizer setup
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(NetworkParameters.policy.learning_rate),
            decay_steps=NetworkParameters.policy.lr_decay_steps,
            decay_rate=NetworkParameters.policy.lr_decay_rate,
            staircase=True,
        )
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # Critic optimizer setup
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(NetworkParameters.state_function_estimator.learning_rate),
            decay_steps=NetworkParameters.state_function_estimator.lr_decay_steps,
            decay_rate=NetworkParameters.state_function_estimator.lr_decay_rate,
            staircase=True,
        )
        state_function_estimator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule
        )
        mean_test_rewards, std_test_rewards = [], []
        self.print_fn(
            f"Training agent for {self.train_iterations} iterations in {self.env_name} environment"
        )

        state = self.env.reset()
        for iteration in tqdm(
            range(
                self.starting_iteration,
                self.starting_iteration + self.train_iterations + 1,
            )
        ):
            (
                states,
                actions,
                value_function_targets,
                done,
                state,
            ) = self.simulate_agent_for_max_timesteps(state)
            if done:
                state = self.env.reset()
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions)
            if tf.is_tensor(value_function_targets[0]):
                value_function_targets = tf.concat(
                    values=value_function_targets, axis=0
                )
            else:
                """
                If using MC returns (if value of n in n-step return greater than episode 
                length), convert floats to tensor and add axis in last dimension to 
                convert shape of targets from (num_transitions,) to (num_transitions,1)
                """
                value_function_targets = tf.expand_dims(
                    tf.convert_to_tensor(value_function_targets, dtype=tf.float32),
                    axis=-1,
                )
            advantage_functions = (
                value_function_targets - self.state_function_estimator(states)
            )
            pi_theta_old, _ = calculate_likelihood(
                self.continuous_action_space, self.policy_net(states), actions
            )
            for _ in range(TrainingParameters.epochs_per_iteration):
                policy_loss, entropy = policy_net_train_step(
                    self.policy_net,
                    policy_optimizer,
                    states,
                    advantage_functions,
                    actions,
                    pi_theta_old,
                    self.entropy_weight,
                    self.pi_ratio
                )
                state_function_estimator_loss = state_function_estimator_train_step(
                    self.state_function_estimator,
                    state_function_estimator_optimizer,
                    states,
                    value_function_targets,
                )
            # Tensorboard logging
            with tensorboard_file_writer.as_default():
                tf.summary.scalar(
                    f"Training loss for {self.policy_net.name}",
                    policy_loss,
                    step=iteration,
                )
                tf.summary.scalar(
                    f"Learning rate for {self.policy_net.name}",
                    policy_optimizer._decayed_lr(tf.float32).numpy(),
                    step=iteration,
                )
                tf.summary.scalar(
                    f"Training loss for {self.state_function_estimator.name}",
                    state_function_estimator_loss,
                    step=iteration,
                )
                tf.summary.scalar(
                    f"Learning rate for {self.state_function_estimator.name}",
                    state_function_estimator_optimizer._decayed_lr(tf.float32).numpy(),
                    step=iteration,
                )
                if entropy:
                    tf.summary.scalar(
                        f"Entropy for action distribution",
                        entropy.numpy(),
                        step=iteration,
                    )
                # Test policy as per test frequency
                if iteration % self.test_frequency == 0:
                    if iteration % self.render_frequency == 0:
                        (
                            mean_test_reward,
                            std_test_reward,
                            max_test_reward,
                            min_test_reward,
                        ) = self.execute_policy(
                            current_training_iteration=iteration,
                            test_episodes=self.test_episodes,
                            render=self.render_decision,
                            save_frequency=TrainingParameters.video_save_frequency,
                        )
                    else:
                        (
                            mean_test_reward,
                            std_test_reward,
                            max_test_reward,
                            min_test_reward,
                        ) = self.execute_policy(
                            test_episodes=self.test_episodes, render=False
                        )
                    self.print_fn(
                        f"After {iteration} training iterations, mean test reward is {mean_test_reward} "
                        f"with std of {std_test_reward} over {self.test_episodes} episodes"
                    )
                    mean_test_rewards.append(mean_test_reward)
                    std_test_rewards.append(std_test_reward)
                    tf.summary.scalar(
                        "Mean rewards over 100 episodes",
                        mean_test_reward,
                        step=iteration,
                    )
                    tf.summary.scalar(
                        "Max rewards over 100 episodes", max_test_reward, step=iteration
                    )
                    tf.summary.scalar(
                        "Min rewards over 100 episodes", min_test_reward, step=iteration
                    )

            # Save model as per model_save_frequency
            if iteration % self.model_save_frequency == 0:
                self.policy_net.save_weights(
                    os.path.join(
                        Directories.output, "saved_models", f"model_{iteration:05}"
                    )
                )
                self.state_function_estimator.save_weights(
                    os.path.join(
                        Directories.output,
                        "saved_models",
                        f"state_function_estimator_{iteration:05}",
                    )
                )

        plt.figure(figsize=(15, 10))
        iterations = list(
            range(
                self.starting_iteration,
                self.starting_iteration + self.train_iterations + 1,
                self.test_frequency,
            )
        )
        plt.errorbar(x=iterations, y=mean_test_rewards, yerr=std_test_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Mean reward with std")
        plt.savefig(os.path.join(Directories.output, "Training_performance.png"))

    def execute_policy(
        self,
        current_training_iteration=None,
        test_episodes=100,
        render=False,
        save_frequency=None,
    ):
        """ Run inference in environment using current policy and evaluate agent performance

        Parameters
        ----------
        current_training_iteration : int, optional
            Current training iteration number for saving progress video, by default None
        test_episodes : int, optional
            Number of test episodes to be simulated, by default 100
        render : bool, optional
            Render environment to stdout (or output window if render is implemented for 
            environment), by default False
        save_frequency : int, optional
            specifies episode number out of test_episodes for which video is to be saved, 
            if None, dont save any video, by default None

        Returns
        -------
        float
            Mean and std of rewards obtained by agent over number of test episodes specified
        """
        if render and save_frequency:

            def video_frequency(x):
                return x % save_frequency == 0

            if self.test_only:
                video_save_path = os.path.join(
                    Directories.output, f"{self.env_name}_test_videos"
                )
            else:
                video_save_path = os.path.join(
                    Directories.output,
                    f"{self.env_name}_Training_progress_videos/{current_training_iteration}",
                )
            self.env = gym.wrappers.Monitor(
                self.env, video_save_path, video_callable=video_frequency, force=True
            )
        rewards = np.zeros(test_episodes)
        for test_episode in range(test_episodes):
            curr_episode_reward, done = 0.0, False
            state = self.env.reset()
            while not done:
                if render:
                    self.env.render()
                if self.continuous_action_space:
                    action = self.policy_net.get_action(
                        tf.expand_dims(state, axis=0), deterministic_policy=False
                    )
                else:
                    action = self.policy_net.get_action(
                        tf.expand_dims(state, axis=0), deterministic_policy=True
                    )
                state, reward, done, _ = self.env.step(action)
                curr_episode_reward += reward
            rewards[test_episode] = curr_episode_reward
        return np.mean(rewards), np.std(rewards), np.max(rewards), np.min(rewards)


def parse_arguments():
    """ Returns command line arguments as per argparse

    Returns
    -------
    argparse arguments
        Parsed arguments from argparse object
    """
    parser = argparse.ArgumentParser(
        description="Policy gradient algorithm implementation: "
        "This can be used to train an agent or test a saved model for policy"
    )
    parser.add_argument(
        "--t",
        dest="test_decision",
        action="store_true",
        help="Test the agent with saved model file",
    )
    parser.add_argument(
        "--m", dest="model_path", default=None, help="Path to model to be used for test"
    )
    parser.add_argument(
        "--mc",
        dest="state_function_estimator_model_path",
        default=None,
        help="Path to advantage estimator model to be used for test",
    )
    parser.add_argument(
        "--e",
        dest="environment_name",
        default="CartPole-v0",
        type=str,
        help="Gym Environment",
    )
    parser.add_argument(
        "--r",
        dest="render_decision",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--ep",
        dest="num_episodes",
        default=None,
        type=int,
        help="Number of training iteration steps and if test flag is given, "
        "this can be used to specify number of test episodes",
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


def run_agent(args, logger=None):
    """ Run agent training or inference

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments provided as input through argparse
    logger : logging.Logger, optional
        logging instance to be used for writing logs to 
        file and stdout, by default None
    """
    with gym.make(args.environment_name) as env:
        agent = PPO(env, args, logger)
        if not args.test_decision:
            agent.train()
            test_episodes = agent.test_episodes
            mean_reward, std_reward, _, _ = agent.execute_policy(
                test_episodes=test_episodes, render=args.render_decision
            )
        else:
            test_episodes = args.num_episodes
            mean_reward, std_reward, _, _ = agent.execute_policy(
                test_episodes=test_episodes,
                render=args.render_decision,
                save_frequency=InferenceParameters.video_save_frequency,
            )
    agent.print_fn(
        f"The reward is {mean_reward} with standard deviation of {std_reward} over "
        f"{test_episodes} episodes"
    )


# Set parameters from config file
config_parameters = parse_config("config/config.yml")
NetworkParameters = config_parameters.Network
TrainingParameters = config_parameters.Training
InferenceParameters = config_parameters.Inference
Directories = config_parameters.Directories

if __name__ == "__main__":
    TIMESTAMP = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args = parse_arguments().parse_args()

    if args.test_decision:
        if args.num_episodes is None:  # Set default test episodes to 100
            args.num_episodes = 100
        if args.model_path is None:
            raise FileNotFoundError("Model path missing for inference")
        # Use saved model path directory
        Directories.output = "/".join(args.model_path.split("/")[:-2])
        logger_file_path = os.path.join(Directories.output, "test.log")
    else:
        if args.num_episodes is None:  # Set default train iterations to 100
            args.num_episodes = 100
        if args.model_path:  # Continue training from pretrained model
            Directories.output = "/".join(args.model_path.split("/")[:-2])
        else:
            Directories.output = os.path.join(Directories.output, TIMESTAMP)
            os.makedirs(Directories.output, exist_ok=True)
        logger_file_path = os.path.join(Directories.output, "train.log")
    logger = setup_logging(logger_file_path)
    run_agent(args, logger)
