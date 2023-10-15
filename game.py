from environment.environment import SnakeEnvironment
from agent.network import DQN
import torch
import numpy as np
import time


if __name__ == "__main__":
    env = SnakeEnvironment(grid_size=6)
    state = env.reset()
    env.render()
    player = int(input("Enter player (0: Human, 1: RL agent): "))
    
    if player == 1:
        
        agent = DQN(env.grid_size, env.action_size)
        model_path = torch.load('snake_agent.pth')
        agent.load_state_dict(model_path)
        agent.eval()

    while not env.done:
        if player == 0:
            action = int(input("Enter action (0: Left, 1: Right, 2: Up, 3: Down): "))
        else:
            with torch.no_grad(): 
                q_values = agent(torch.tensor(state, dtype=torch.float32))
                action = np.argmax(q_values.numpy())
    
        next_state, reward, done = env.step(action)
        env.render()
        state = next_state
        time.sleep(0.3)

    print("Game Over. Score:", env.score)