import gym
import matplotlib.pyplot as plt
import numpy as np


def evaluate_policy(ValueFunction, Policy=None, policy_iteration=True, asynchronous=False):
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
                Q_value_action += TransitionProb * (Reward + gamma * OriginalValueFunction[NewState])
            """
            Verify if action probability is to be multiplied for value iteration
            Q_values_state.append(Policy[state, action] * Q_value_action)
            """
            if policy_iteration:
                Q_values_state.append(Policy[state, action] * Q_value_action)
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


def improve_policy(ValueFunction, Policy):
    OriginalPolicy = Policy.copy()
    for state in range(env.observation_space.n):
        StateQValues = []
        for action in range(env.action_space.n):
            TransitionTuplesList = env.P[state][action]
            Q_value_action = 0
            for TransitionProb, NewState, Reward, _ in TransitionTuplesList:
                Q_value_action += (Reward + gamma * TransitionProb * ValueFunction[NewState])
            StateQValues.append(Q_value_action)
            Policy[state, action] = 0
        Policy[state, np.argmax(StateQValues)] = 1
    return Policy, (Policy == OriginalPolicy).all()


def greedy_policy(Q_values):
    return np.argmax(Q_values)


def test(Policy):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = greedy_policy(Policy[state, :])
        env.render()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(total_reward)


def print_policy(Policy):
    num_states_single_line = env.desc.shape[0]
    grid_data = np.zeros((num_states_single_line, num_states_single_line))
    for i in range(num_states_single_line):
        for j in range(num_states_single_line):
            if env.desc[i, j] == b'S':
                grid_data[i, j] = 1
            elif env.desc[i, j] == b'F':
                grid_data[i, j] = 0.2
            elif env.desc[i, j] == b'G':
                grid_data[i, j] = 0.7
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    plt.imshow(grid_data, cmap=plt.get_cmap('gnuplot'))
    for i in range(num_states_single_line):
        for j in range(num_states_single_line):
            info = f'{action_names[greedy_policy(Policy[(i * num_states_single_line) + j, :])]}\n({env.desc[i, j].astype("<U1")})'
            ax.text(j, i, info, ha='center', va='center', backgroundcolor='c')
    plt.title('Greedy policy action in each state')
    plt.show()


def run_PI():
    ValueFunction = np.zeros(env.observation_space.n)
    Policy = np.zeros((env.observation_space.n, env.action_space.n))
    np.put_along_axis(arr=Policy, indices=np.random.randint(0, env.action_space.n, (env.observation_space.n, 1)),
                      values=1, axis=1)
    PolicyConverged = False
    while not PolicyConverged:
        # Policy Evaluation
        num_iterations = 0
        MaxUpdate = tolerance + 1
        while (MaxUpdate > tolerance) and (num_iterations < max_iterations):
            UpdatedValueFunction = evaluate_policy(ValueFunction.copy(), Policy)
            MaxUpdate = np.max(np.abs(UpdatedValueFunction - ValueFunction))
            ValueFunction = UpdatedValueFunction
            num_iterations += 1
        # Policy Improvement
        Policy, PolicyConverged = improve_policy(ValueFunction, Policy)
    test(Policy)
    print_policy(Policy)


def run_VI():
    ValueFunction = np.zeros(env.observation_space.n)
    Policy = np.zeros((env.observation_space.n, env.action_space.n))
    MaxUpdate = tolerance + 1
    num_iterations = 0
    while MaxUpdate > tolerance:
        UpdatedValueFunction = evaluate_policy(ValueFunction.copy(), policy_iteration=False)
        MaxUpdate = np.max(np.abs(ValueFunction - UpdatedValueFunction))
        ValueFunction = UpdatedValueFunction
        num_iterations += 1
        if num_iterations == max_iterations:
            raise TimeoutError(f"Value Function failed to converge after {max_iterations} iterations")

    Policy, _ = improve_policy(ValueFunction, Policy)

    test(Policy)
    print_policy(Policy)

def setup_logging():
    pass

def parse_arguments():
    pass

if __name__ == '__main__':
    action_names = {0: "LEFT",
                    1: "DOWN",
                    2: "RIGHT",
                    3: "UP"}
    with gym.make('FrozenLake-v0', is_slippery=False) as env:
        gamma = 0.9
        max_iterations = 10000
        tolerance = 0.1
        run_VI()
