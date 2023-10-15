import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

from agent.network import DQN
from agent.replay_buffer import ReplayBuffer
from environment.environment import SnakeEnvironment

BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPS_MIN= 0.01
EPS_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 10


if __name__ == "__main__":

    # Initialize the environment and hyperparameters
    env = SnakeEnvironment(grid_size=6)  # You'll need to define your Snake environment

    state_size = env.state_size
    action_size = env.action_size
    grid_size = env.grid_size

    # Initialize the DQN and target DQN
    # dqn = DQN(grid_size, action_size)
    # target_dqn = DQN(grid_size, action_size)
    # target_dqn.load_state_dict(dqn.state_dict())  # Initialize target DQN with the same weights

    dqn = DQN(grid_size, action_size)
    # load from checkpoint
    dqn.load_state_dict(torch.load('snake_agent.pth'))
    target_dqn = DQN(grid_size, action_size)
    target_dqn.load_state_dict(dqn.state_dict()) 

    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)

    # Training loop
    episode_rewards = []

    for episode in range(1002):
        state = env.reset()
        total_reward = 0

        while True:
            # Epsilon-greedy exploration
            if random.uniform(0, 1) < EPSILON:
                action = env.random_action()
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32))
                    action = np.argmax(q_values.numpy())

            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer.buffer) > BATCH_SIZE:
                # Sample a batch of experiences from the replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.int64)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)


                # Calculate Q-values
                q_values = dqn(states)
                next_q_values = target_dqn(next_states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values
                target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values.max(1)[0]


                # Compute loss and backpropagate
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target DQN periodically
                if episode % TARGET_UPDATE_FREQUENCY == 0:
                    target_dqn.load_state_dict(dqn.state_dict())

            if done:
                break

        # Decay epsilon
        EPSILON = max(EPSILON * EPS_DECAY, EPS_MIN)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}")

        if (episode + 1) % 100 == 0:
            checkpoint_name = f'models\\snake_agent_episode_{episode + 1}.pth'
            torch.save(target_dqn.state_dict(), checkpoint_name)

    # save the agent
    torch.save(target_dqn.state_dict(), 'snake_agent.pth')

    # Plot episode rewards
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()
