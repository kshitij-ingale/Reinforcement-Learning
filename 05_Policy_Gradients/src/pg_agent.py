""" Script implementing Policy gradients agent (REINFORCE and actor critic) for Gym environment """
import argparse, gym, os, time, logging, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from network import Net, policy_net_train_step, state_function_estimator_train_step

np.random.seed(42)
from tqdm import tqdm
import sys

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
from utils.utils import parse_config


class PolicyGrad:
    def __init__(self, env, parameters, logger=None):
        # Gym environment parameters
        self.env_name = parameters.environment_name
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.num_actions = self.env.action_space.n
        else:
            raise NotImplementedError("Continuous actions not yet supported")
        # Policy network or Actor network in AC
        if TrainingParameters.use_REINFORCE:
            self.policy_net = Net(
                self.state_dim, self.num_actions, NetworkParameters.policy, "policy_net"
            )
        else:
            self.policy_net = Net(
                self.state_dim, self.num_actions, NetworkParameters.policy, "actor"
            )
        self.use_reinforce = TrainingParameters.use_REINFORCE
        self.render_decision = parameters.render_decision
        self.test_only = parameters.test_decision
        if logger is not None:
            self.logger = logger
            self.print_fn = self.logger.info
        else:
            self.print_fn = print
        if not self.test_only:
            # Training parameters
            self.train_episodes = parameters.num_episodes
            self.test_episodes = TrainingParameters.test_episodes
            self.test_frequency = TrainingParameters.test_frequency
            self.render_frequency = TrainingParameters.render_frequency
            self.video_save_frequency = TrainingParameters.video_save_frequency
            self.model_save_frequency = TrainingParameters.model_save_frequency
            self.discount = TrainingParameters.discount
            self.starting_episode = 0
            if TrainingParameters.use_baseline:
                if TrainingParameters.use_REINFORCE:
                    self.state_function_estimator = Net(
                        self.state_dim,
                        1,
                        NetworkParameters.state_function_estimator,
                        "baseline_net",
                    )
                    self.print_fn(
                        "Initiating model for REINFORCE with baseline algorithm"
                    )
                else:
                    self.state_function_estimator = Net(
                        self.state_dim,
                        1,
                        NetworkParameters.state_function_estimator,
                        "critic",
                    )
                    self.print_fn("Initiating model for actor-critic algorithm")
            else:
                self.print_fn(
                    "Initiating model for REINFORCE (without baseline) algorithm"
                )

        if parameters.model_path:
            self.policy_net.load_weights(parameters.model_path)
            self.print_fn(
                f"Loading weights for {self.policy_net.name} from {parameters.model_path}"
            )
            self.starting_episode = int(parameters.model_path[-5:])
            if parameters.state_function_estimator_model_path:
                self.state_function_estimator.load_weights(
                    parameters.state_function_estimator_model_path
                )
                self.print_fn(
                    f"Loading weights for {self.state_function_estimator.name} "
                    f"from {parameters.state_function_estimator_model_path}"
                )

    def simulate_episode_for_reinforce(self):
        states, actions, rewards = [], [], []
        done = False
        state = self.env.reset()
        while not done:
            action = tf.random.categorical(
                self.policy_net(tf.expand_dims(state, axis=0)), 1
            ).numpy()[0][0]
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards

    @staticmethod
    def get_discounted_returns(rewards, discount):
        discounted_rewards = np.zeros((len(rewards), 1), dtype=np.float32)
        discounted_rewards[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            discounted_rewards[i] = discount * discounted_rewards[i + 1] + rewards[i]
        return discounted_rewards

    def simulate_episode_for_ac(self):
        states, actions, rewards, value_function_targets = [], [], [], []
        done = False
        state = self.env.reset()
        current_time_step, Terminal_time_step = 0, float("inf")
        bootstrap_discounted_return = None
        while True:
            if current_time_step < Terminal_time_step:
                action = tf.random.categorical(
                    self.policy_net(tf.expand_dims(state, axis=0)), 1
                ).numpy()[0][0]
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    Terminal_time_step = current_time_step + 1
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            update_time_step = current_time_step - TrainingParameters.n_step_bootstrap
            if update_time_step >= 0:
                simulated_discounted_return = 0.0
                for time_step in range(
                    update_time_step,
                    min(
                        update_time_step + TrainingParameters.n_step_bootstrap,
                        Terminal_time_step,
                    ),
                ):
                    simulated_discounted_return += (
                        self.discount ** (time_step - update_time_step)
                    ) * rewards[time_step]
                if (
                    update_time_step + TrainingParameters.n_step_bootstrap
                    < Terminal_time_step
                ):
                    bootstrap_state = states[
                        update_time_step + TrainingParameters.n_step_bootstrap
                    ]
                    bootstrap_discounted_return = (
                        self.discount ** TrainingParameters.n_step_bootstrap
                    ) * self.state_function_estimator(
                        tf.expand_dims(bootstrap_state, axis=0)
                    )
                value_function_action = simulated_discounted_return
                if bootstrap_discounted_return:
                    value_function_action += bootstrap_discounted_return
                value_function_targets.append(value_function_action)
            current_time_step += 1
            if update_time_step == Terminal_time_step - 1:
                break
        return states, actions, value_function_targets

    def train(self):
        tensorboard_dir = os.path.join(Directories.output, "tensorboard")
        tensorboard_file_writer = tf.summary.create_file_writer(tensorboard_dir)
        if TrainingParameters.use_baseline:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                float(NetworkParameters.state_function_estimator.learning_rate),
                decay_steps=1000,
                decay_rate=0.5,
                staircase=True,
            )
            state_function_estimator_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule
            )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(NetworkParameters.policy.learning_rate),
            decay_steps=1000,
            decay_rate=0.5,
            staircase=True,
        )
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        mean_test_rewards, std_test_rewards = [], []
        self.print_fn(
            f"Training agent for {self.train_episodes} episodes in {self.env_name} environment"
        )

        for episode in tqdm(
            range(
                self.starting_episode, self.starting_episode + self.train_episodes + 1
            )
        ):
            if self.use_reinforce:
                states, actions, rewards = self.simulate_episode_for_reinforce()
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions)
                discounted_returns = tf.convert_to_tensor(
                    self.get_discounted_returns(rewards, self.discount)
                )
                if TrainingParameters.use_baseline:
                    estimated_value_functions = self.state_function_estimator(states)
                    advantage_functions = discounted_returns - estimated_value_functions
                    policy_loss = policy_net_train_step(
                        self.policy_net,
                        policy_optimizer,
                        states,
                        advantage_functions,
                        actions,
                    )
                    state_function_estimator_loss = state_function_estimator_train_step(
                        self.state_function_estimator,
                        state_function_estimator_optimizer,
                        states,
                        discounted_returns,
                    )
                else:
                    policy_loss = policy_net_train_step(
                        self.policy_net,
                        policy_optimizer,
                        states,
                        discounted_returns,
                        actions,
                    )
            else:
                states, actions, value_function_targets = self.simulate_episode_for_ac()
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions)
                if tf.is_tensor(value_function_targets[0]):
                    value_function_targets = tf.concat(values=value_function_targets, axis=0)
                else:
                    """
                    If using MC returns (using high value of n in n-step return), convert floats 
                    to tensor and add axis in last dimension to convert shape of targets from 
                    (num_transitions,) to (num_transitions,1)
                    """
                    value_function_targets = tf.expand_dims(
                        tf.convert_to_tensor(value_function_targets, dtype=tf.float32),
                        axis=-1,
                    )

                advantage_functions = (
                    value_function_targets - self.state_function_estimator(states)
                )
                policy_loss = policy_net_train_step(
                    self.policy_net,
                    policy_optimizer,
                    states,
                    advantage_functions,
                    actions,
                )
                state_function_estimator_loss = state_function_estimator_train_step(
                    self.state_function_estimator,
                    state_function_estimator_optimizer,
                    states,
                    value_function_targets,
                )

            with tensorboard_file_writer.as_default():
                tf.summary.scalar(
                    f"Training loss for {self.policy_net.name}",
                    policy_loss,
                    step=episode,
                )
                tf.summary.scalar(
                    f"Learning rate for {self.policy_net.name}",
                    policy_optimizer._decayed_lr(tf.float32).numpy(),
                    step=episode,
                )
                if TrainingParameters.use_baseline:
                    tf.summary.scalar(
                        f"Training loss for {self.state_function_estimator.name}",
                        state_function_estimator_loss,
                        step=episode,
                    )
                    tf.summary.scalar(
                        f"Learning rate for {self.state_function_estimator.name}",
                        state_function_estimator_optimizer._decayed_lr(
                            tf.float32
                        ).numpy(),
                        step=episode,
                    )
                # Test policy as per test frequency
                if episode % self.test_frequency == 0:
                    if episode % self.render_frequency == 0:
                        mean_test_reward, std_test_reward = self.execute_policy(
                            current_training_episode=episode,
                            test_episodes=self.test_episodes,
                            render=self.render_decision,
                            save_frequency=TrainingParameters.video_save_frequency,
                        )
                    else:
                        mean_test_reward, std_test_reward = self.execute_policy(
                            test_episodes=self.test_episodes, render=False
                        )
                    self.print_fn(
                        f"After {episode} episodes, mean test reward is {mean_test_reward} "
                        f"with std of {std_test_reward} over {self.test_episodes} episodes"
                    )
                    mean_test_rewards.append(mean_test_reward)
                    std_test_rewards.append(std_test_reward)
                    tf.summary.scalar(
                        "Mean rewards over 100 episodes", mean_test_reward, step=episode
                    )

            # Save model as per model_save_frequency
            if episode % self.model_save_frequency == 0:
                self.policy_net.save_weights(
                    os.path.join(
                        Directories.output, "saved_models", f"model_{episode:05}"
                    )
                )
                if TrainingParameters.use_baseline:
                    self.state_function_estimator.save_weights(
                        os.path.join(
                            Directories.output,
                            "saved_models",
                            f"state_function_estimator_{episode:05}",
                        )
                    )

        plt.figure(figsize=(15, 10))
        episodes = list(
            range(0, self.train_episodes + 1, TrainingParameters.test_frequency)
        )
        plt.errorbar(x=episodes, y=mean_test_rewards, yerr=std_test_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Mean reward with std")
        plt.savefig(os.path.join(Directories.output, "Training_performance.png"))

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
                video_save_path = os.path.join(
                    Directories.output, f"{self.env}_test_videos"
                )
            else:
                video_save_path = os.path.join(
                    Directories.output,
                    f"{self.env}_Training_progress_videos/{current_training_episode}",
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
                # Reshape state to add batch dimension
                action_probabilities = self.policy_net.get_action_probabilities(
                    tf.expand_dims(state, axis=0)
                )
                action = np.argmax(action_probabilities)
                state, reward, done, _ = self.env.step(action)
                curr_episode_reward += reward
            rewards[test_episode] = curr_episode_reward
        return np.mean(rewards), np.std(rewards)


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
        help="Number of episodes for training and if test flag is given, "
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

    with gym.make(args.environment_name) as env:
        agent = PolicyGrad(env, args, logger)
        if not args.test_decision:
            agent.train()
            test_episodes = agent.test_episodes
            mean_reward, std_reward = agent.execute_policy(
                test_episodes=test_episodes, render=args.render_decision
            )
        else:
            test_episodes = args.num_episodes
            mean_reward, std_reward = agent.execute_policy(
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
# If using Actor critic, set use_baseline to True for critic model
if not TrainingParameters.use_REINFORCE:
    TrainingParameters.use_baseline = True
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
        if args.num_episodes is None:  # Set default train episodes to 5000
            args.num_episodes = 5000
        if args.model_path:  # Continue training from pretrained model
            Directories.output = "/".join(args.model_path.split("/")[:-2])
        else:
            Directories.output = os.path.join(Directories.output, TIMESTAMP)
            os.makedirs(Directories.output, exist_ok=True)
        logger_file_path = os.path.join(Directories.output, "train.log")
    logger = setup_logging(logger_file_path)
    run_agent(args, logger)
