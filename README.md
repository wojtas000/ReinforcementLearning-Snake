# Snake

This is an attempt to implement agent playing simple Snake game.

# Agent

Agent is a CNN-based model. It is supported with reinforcement learning to study the optimal behavior in the game.

# Environment

1. States - state of the game is a  $n \times n$ grid, containing information about position of snake's head, whole tail and food of the board.
2. Actions - for each state, 4 unique actions are possible: to move up, down, left or right
3. Rewards - snake is rewarded for collecting food, having longer tail and coming closer to the next food (to encourage moving towards the food and not getting stuck)
4. Punishments - snake is punished for running into the border of grid, eating its own tail and for time ellapsed from eating the previous food (the longer the snake is not collecting the food, the more it will be punished)
