
# Reinforcement Learning for OpenAI Gym Environment

Function approximation can be used to compute Q-values to determine optimal policy for agents. This implementation focuses on DQN algorithm as described by Mnih et al. ([https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)) originally prescribed for Atari games. Other variants of DQN like double DQN ([https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)) and dueling networks ([https://arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581)) were explored as well.
For this implementation, OpenAI Gym environments of CartPole and MountainCar were considered for implementation. The training was performed for 10000 episodes and checkpoint results were obtained.

In addition to this, REINFORCE and N-step Actor Critic algorithms were implemented for Lunarlander environment of OpenAI Gym

## Repository Structure:
- **videos** : Directory for saving videos indicating training progress
- **DQN_Implementation.py** : Script for training the agent with DQN algorithm
- **Double_DQN.py** : Script for training the agent with double DQN algorithm
- **Duel_Network.py** : Script for training the agent with dueling network architecture
- **a2c.py** : Script for training the agent with N-step A2C algorithm (Lunar-lander)
- **reinforce.py** : Script for training the agent with REINFORCE architecture (Lunar-lander)

## Results:
Following is the training progress for mountaincar environment using DQN.

After training for 3333 episodes
![](https://github.com/kshitij-ingale/Reinforcement-Learning/blob/master/Videos/Mountain%20Car/1.gif)

After training for 6666 episodes
![](https://github.com/kshitij-ingale/Reinforcement-Learning/blob/master/Videos/Mountain%20Car/2.gif)

After training for 9999 episodes
![](https://github.com/kshitij-ingale/Reinforcement-Learning/blob/master/Videos/Mountain%20Car/3.gif)
