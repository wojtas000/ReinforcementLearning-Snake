import numpy as np
import random
import matplotlib.pyplot as plt

class SnakeEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_size = 4
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.snake = [(0, 0)]  # Snake starts at the top-left corner
        self.head = self.snake[0]
        self.food = self.generate_food()
        self.direction = (1, 0)  # Initial direction: right
        self.done = False
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size  # Maximum steps before game over
        self.move_counter_from_last_eat = 0

        # Place the snake and food on the grid
        self.update_grid()

        # Return the current state as a flattened grid
        return self.get_state()

    def generate_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for x, y in self.snake:
            self.grid[y][x] = 1  # Mark snake body as 1
        x, y = self.food
        self.grid[y][x] = 2  # Mark food as 2

    def get_state(self):
        return self.grid.flatten()

    def calculate_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        if self.done:
            raise ValueError("Game over. Call reset() to restart the game.")

        if action == 0:  # Left
            new_direction = (-1, 0)
        elif action == 1:  # Right
            new_direction = (1, 0)
        elif action == 2:  # Up
            new_direction = (0, -1)
        elif action == 3:  # Down
            new_direction = (0, 1)
        else:
            raise ValueError("Invalid action.")

        # Check if the new direction is the opposite of the current direction (illegal move)
        if (new_direction[0] + self.direction[0], new_direction[1] + self.direction[1]) == (0, 0):
            new_direction = self.direction  # Keep moving in the same direction

        new_head = (self.head[0] + new_direction[0], self.head[1] + new_direction[1])

        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
            or new_head in self.snake
        ):
            # Game over due to collision with wall or itself
            self.done = True
            reward = -10
        else:
            self.snake.insert(0, new_head)  # Move the head
            self.head = new_head
            self.direction = new_direction

            if self.head == self.food:
                # Snake eats the food
                self.food = self.generate_food()
                self.move_counter_from_last_eat = 0
                self.score += 1
                reward = self.score + 1
            else:
                # Snake moves without eating
                removed_tail = self.snake.pop()
                self.grid[removed_tail[1]][removed_tail[0]] = 0  # Clear the tail
                self.move_counter_from_last_eat += 1
                distance_to_food = self.calculate_distance(self.head, self.food)
                # final reward should increase as the snake gets closer to the food and decrease with the number of steps since the last eat
                reward = 0
                

            self.update_grid()
            self.steps += 1

            if self.steps >= self.max_steps:
                # Game over due to maximum steps reached
                self.done = True

        return self.get_state(), reward, self.done

    def random_action(self):
        return random.randint(0, 3)  # Random action (0: Left, 1: Right, 2: Up, 3: Down)

    def render(self):
        plt.clf()
        plt.imshow(self.grid, cmap='viridis', origin='upper')

        # Add grid lines
        for x in range(self.grid_size + 1):
            plt.axvline(x - 0.5, color='white', linewidth=0.5)
            plt.axhline(x - 0.5, color='white', linewidth=0.5)

        plt.xticks([])
        plt.yticks([])  
        plt.pause(0.1)  

# Example usage:
if __name__ == "__main__":
    env = SnakeEnvironment(grid_size=5)  # Create a Snake environment with a 5x5 grid
    state = env.reset()
    env.render()
    
    while not env.done:
        action = int(input("Enter action (0: Left, 1: Right, 2: Up, 3: Down): "))
        # action = env.random_action()  # Replace with your RL agent's action
        next_state, reward, done = env.step(action)
        env.render()

    print("Game Over. Score:", env.score)
