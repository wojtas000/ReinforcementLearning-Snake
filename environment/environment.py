import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Set

class SurvivalEnvironment:
    def __init__(self, grid_size: int = 5) -> None:
        self.grid_size: int = grid_size
        self.state_size: int = grid_size * grid_size
        self.action_size: int = 4
        self.reset()

    def random_action(self) -> int:
        '''
        Returns a random action from the action space
        '''
        return random.randint(0, self.action_size - 1)

    def get_direction(self, action: int) -> Tuple[int, int]:
        '''
        Returns the direction of the action
        '''
        if action == 0:  # Left
            new_direction: Tuple[int, int] = (-1, 0)
        elif action == 1:  # Right
            new_direction = (1, 0)
        elif action == 2:  # Up
            new_direction = (0, -1)
        elif action == 3:  # Down
            new_direction = (0, 1)
        else:
            raise ValueError("Invalid action.")
        
        return new_direction

    def reset(self) -> List[int]:
        '''
        Resets the environment to the initial state
        '''
        self.grid: List[List[int]] = np.zeros((self.grid_size, self.grid_size), dtype=int).tolist()
        self.snake: List[Tuple[int, int]] = [(0, 0)]
        self.head: Tuple[int, int] = self.snake[0]
        self.food: Tuple[int, int] = self.generate_food()
        self.direction: Tuple[int, int] = (1, 0)  # Initial direction: right
        self.done: bool = False
        self.score: int = 0
        self.steps: int = 0
        self.max_steps: int = self.grid_size * self.grid_size * 2
        self.move_counter_from_last_eat: int = 0
        self.visited_positions: Set[Tuple[int, int]] = set()

        self.update_grid()
        return self.get_state()

    def generate_food(self) -> Tuple[int, int]:
        '''
        Generates a new food position that is not on the snake
        '''
        while True:
            food: Tuple[int, int] = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def update_grid(self) -> None:
        '''
        Updates the grid with the current snake and food positions
        '''
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int).tolist()
        for x, y in self.snake:
            self.grid[y][x] = 1
            self.visited_positions.add((x, y))
        x, y = self.food
        self.grid[y][x] = 2

    def get_state(self) -> List[int]:
        '''
        Returns the current state as a flattened grid
        '''
        return [cell for row in self.grid for cell in row]

    def calculate_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        '''
        Calculates the Manhattan distance between two points
        '''
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def collision_with_wall(self, new_head: Tuple[int, int]) -> bool:
        '''
        Checks if the new head position collides with the wall
        '''
        return (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        )
    
    def collision_with_self(self, new_head: Tuple[int, int]) -> bool:
        '''
        Checks if the new head position collides with the snake body
        '''
        return new_head in self.snake

    def direction_opposite_to_current(self, direction: Tuple[int, int]) -> bool:
        '''
        Checks if the new direction is opposite to the current direction
        '''
        return (direction[0] + self.direction[0], direction[1] + self.direction[1]) == (0, 0)

    def move_head(self, new_head: Tuple[int, int]) -> None:
        '''
        Moves the snake head to the new position
        '''
        self.snake.insert(0, new_head)
        self.head = new_head

    def step(self, action: int) -> Tuple[List[int], int, bool]:
        '''
        Moves the snake in the given direction and returns the new state, reward, and done flag
        '''
        if self.done:
            raise ValueError("Game over. Call reset() to restart the game.")

        new_direction: Tuple[int, int] = self.get_direction(action)

        if self.direction_opposite_to_current(new_direction):
            new_direction = self.direction 

        new_head: Tuple[int, int] = (self.head[0] + new_direction[0], self.head[1] + new_direction[1])

        if self.collision_with_wall(new_head) or self.collision_with_self(new_head):
            self.done = True
            reward: int = -10
        else:
            self.move_head(new_head)
            self.direction = new_direction

            if self.head == self.food:
                self.food = self.generate_food()
                self.score += 1
                reward = self.exploration_reward() + self.staying_alive_reward()
            else:
                self.clear_tail()
                reward = self.exploration_reward() + self.staying_alive_reward()

            self.update_grid()
            self.steps += 1

            if self.steps >= self.max_steps:
                self.done = True

        return self.get_state(), reward, self.done
    
    def render(self) -> None:
        '''
        Renders the current state of the environment
        '''
        plt.clf()
        plt.imshow(self.grid, cmap='viridis', origin='upper')

        # Add grid lines
        for x in range(self.grid_size + 1):
            plt.axvline(x - 0.5, color='white', linewidth=0.5)
            plt.axhline(x - 0.5, color='white', linewidth=0.5)

        plt.xticks([])
        plt.yticks([])  
        plt.pause(0.1)  

    def exploration_reward(self) -> int:
        '''
        Returns a reward for exploring a new position on the grid
        '''
        if self.head not in self.visited_positions:
            return 1
        else:
            return 0
        
    def staying_alive_reward(self) -> float:
        '''
        Returns a reward for staying alive
        '''
        return 1 / self.max_steps
    
    def clear_tail(self) -> None:
        '''
        Removes the tail of the snake
        '''
        removed_tail: Tuple[int, int] = self.snake.pop()
        self.grid[removed_tail[1]][removed_tail[0]] = 0

class EatAndGrowEnvironment(SurvivalEnvironment):
    def step(self, action: int) -> Tuple[List[int], int, bool]:
        state, reward, done = super().step(action)

        # Additional reward for eating food and a penalty for not eating
        if self.head == self.food:
            reward += 10  # Reward for eating food
        else:
            # calculate distance to food
            distance_to_food: int = self.calculate_distance(self.head, self.food)
            reward -= distance_to_food / self.grid_size ** 2

        return state, reward, done

class FoodEnvironment(SurvivalEnvironment):
    def __init__(self, grid_size: int = 5) -> None:
        super().__init__(grid_size)
        
    def step(self, action: int) -> Tuple[List[int], int, bool]:
        '''
        Moves the snake in the given direction and returns the new state, reward, and done flag
        '''
        if self.done:
            raise ValueError("Game over. Call reset() to restart the game.")

        new_direction: Tuple[int, int] = self.get_direction(action)

        if self.direction_opposite_to_current(new_direction):
            new_direction = self.direction 

        new_head: Tuple[int, int] = (self.head[0] + new_direction[0], self.head[1] + new_direction[1])

        if self.collision_with_wall(new_head) or self.collision_with_self(new_head):
            self.done = True
            reward: int = -10
        else:
            self.move_head(new_head)
            self.direction = new_direction
            distance_to_food: int = self.calculate_distance(self.head, self.food)

            if self.head == self.food:
                self.food = self.generate_food()
                self.score += 1
                reward = 10 * self.score - distance_to_food / self.grid_size ** 2
            else:
                self.clear_tail()
                reward = - distance_to_food / self.grid_size ** 2

            self.update_grid()
            self.steps += 1

            if self.steps >= self.max_steps:
                self.done = True

        return self.get_state(), reward, self.done

# Example usage:
if __name__ == "__main__":
    # Train the agent to stay alive
    survival_env = SurvivalEnvironment(grid_size=5)
    state = survival_env.reset()

    while not survival_env.done:
        action = survival_env.random_action()
        next_state, reward, done = survival_env.step(action)

    print("Survival Game Over. Score:", survival_env.score)

    # Train the agent to eat and grow
    eat_and_grow_env = EatAndGrowEnvironment(grid_size=5)
    state = eat_and_grow_env.reset()

    while not eat_and_grow_env.done:
        action = eat_and_grow_env.random_action()
        next_state, reward, done = eat_and_grow_env.step(action)

    print("Eat and Grow Game Over. Score:", eat_and_grow_env.score)
