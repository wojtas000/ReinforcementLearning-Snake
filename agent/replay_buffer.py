from typing import Any, Tuple
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Replay buffer for storing and sampling experiences.

        Parameters:
            capacity (int): Maximum capacity of the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state: Any, action: Any, reward: Any, next_state: Any, done: bool) -> None:
        """
        Add an experience to the replay buffer.

        Parameters:
            state (Any): Current state.
            action (Any): Action taken.
            reward (Any): Reward received.
            next_state (Any): Next state.
            done (bool): Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the replay buffer.

        Parameters:
            batch_size (int): Size of the batch to sample.

        Returns:
            Tuple: Tuple of (state, action, reward, next_state, done) for the sampled batch.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
