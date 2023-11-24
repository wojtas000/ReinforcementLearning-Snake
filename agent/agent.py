import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

from agent.network import DQN
from agent.replay_buffer import ReplayBuffer
from environment.environment import SurvivalEnvironment, EatAndGrowEnvironment

BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 10
NUM_EPISODES = 10002
WINDOW_SIZE = 20

# Set up interactive plotting
plt.ion()
fig, ax = plt.subplots()
episode_rewards = []

if __name__ == "__main__":
    # Initialize the environment and hyperparameters
    env = SurvivalEnvironment(grid_size=6) 
    state_size = env.state_size
    action_size = env.action_size
    grid_size = env.grid_size
    dqn = DQN(grid_size, action_size)
    # dqn.load_state_dict(torch.load('snake_agent.pth'))
    target_dqn = DQN(grid_size, action_size)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=10000)

    # Training loop
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
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
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.int64)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)

                q_values = dqn(states)
                next_q_values = target_dqn(next_states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values.max(1)[0]

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if episode % TARGET_UPDATE_FREQUENCY == 0:
                    target_dqn.load_state_dict(dqn.state_dict())

            if done or env.steps >= env.max_steps:
                break

        EPSILON = max(EPSILON * EPS_DECAY, EPS_MIN)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}")

        if (episode + 1) % 1000 == 0:
            checkpoint_name = f'models\\snake_agent_episode_{episode + 1}.pth'
            torch.save(target_dqn.state_dict(), checkpoint_name)

        # if episode == NUM_EPISODES // 2:
        #     env = EatAndGrowEnvironment(grid_size=6)

        # Calculate and plot the rolling average
        if len(episode_rewards) >= WINDOW_SIZE:
            rolling_avg = np.convolve(episode_rewards, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
            ax.clear()
            ax.plot(rolling_avg)
            ax.set(xlabel="Episode", ylabel=f"Total Reward (Moving Average - Window Size: {WINDOW_SIZE})", title="Training Rewards")
            fig.canvas.draw()
            fig.canvas.flush_events()

    # Save the agent
    torch.save(target_dqn.state_dict(), 'snake_agent.pth')

    # Final plot
    plt.ioff()
    plt.plot(rolling_avg)
    plt.xlabel("Episode")
    plt.ylabel(f"Total Reward (Moving Average - Window Size: {WINDOW_SIZE})")
    plt.title("Training Rewards")
    plt.show()
