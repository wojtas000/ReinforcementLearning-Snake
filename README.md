# Snake

This is an attempt to implement agent playing simple Snake game.

# Agent

Agent is a CNN-based model. It is supported with reinforcement learning to study the optimal behavior in the game.

# Environment

1. States - state of the game is a  $n \times n$ grid, containing information about position of snake's head, whole tail and food of the board.
2. Actions - for each state, 4 unique actions are possible: to move up, down, left or right
3. Rewards - snake is rewarded for collecting food, for staying alive and for exploring the board.
4. Punishments - snake is punished for running into the border of grid, eating its own tail and for running away from food.

# Training 

The process of training the snake agent consists of two major steps:
1. Survival + map exploring - In this stage the training process is aimed at making snake live as long as possible and explore the whole board. 
2. Eating food + growing bigger - In this stage, the snake agent is rewarded for eating food (proportionally to its length. Therefore the longer the snake, the more encouragement it gets for eating more food, as it is also more dangerous and the snake could potentially try to avoid growing bigger). The snake is punished for not coming closer to food, to discourage him from running away from food.


![Training process](https://github.com/wojtas000/Snake/blob/main/plots/snake_training.png)
