# Take It Easy

 - Simple python implementation of the game
 - [Naïve heuristic based on probabilities](#Bots)
 - [MCTS based solver](#Bots)
 - [Neural Network based solver using DQN](#Bots)

## Game

[Take It Easy](https://en.wikipedia.org/wiki/Take_It_Easy_(game)) is a single player strategy board game involving a hexagonal board with 19 tiles and 27 pieces.

The player draws a random piece from their stack and places it onto the board, repeat until full.

![Half-filled Take It Easy grid.](.github/header.png)

The final score is then calculated as follows: for every completed line, the player gets `line_length * line_value` points. The goal is to get the highest possible score.

![Score being calculated](.github/scoring.png)

The score for the above example is `121` points.

The perfect score is exactly `307`, there exist 16 such combinations.

Due to the large number of possible combinations of drawn pieces, different drawing orders and placements, the game is not fully solvable.

$$ \frac{27!}{(27 - 19)!} \cdot 19! \approx 3.2 × 10^{40} \quad \text{possible games} $$

## Bots

Due to the computational impossibility of computing all possible games, we have to make due with heuristics.

I implemented multiple solutions:
 1. A neural-network implemented as inspired/copied from [polarbart/TakeItEasyAI](https://github.com/polarbart/TakeItEasyAI)
 2. Monte-Carlo Tree Search
 3. Simple heuristic that uses the amount of pieces that are left and the probability of drawing each piece

