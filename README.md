# Take It Easy

 - `python` implementation of Take it Easy with a simple API and support for generating visualisations of the board (as HTML)
 - Optimised `rust` bindings for `python` that implement a faster, optimised version.
 - [Pre-built wheels to train your own model on Google Colab](#train-a-model-yourself)

See the [rules](#game), how to [use the library](#usage) and the details about the following [bots](#bots) and approaches to "solving" the game I implemented:
 - [Naïve heuristic based on probabilities](#bots)
 - [MCTS based solver](#bots)
 - [Neural Network based solver using DQN](#neural-network)

If you want to learn more about the models and training, check out the [detailed training logs](./models/README.md).

## Game

If you want to try out the game, go to [take-it-easy.obrhubr.org](https://take-it-easy.obrhubr.org).

[Take It Easy](https://en.wikipedia.org/wiki/Take_It_Easy_(game)) is a single player strategy board game involving a hexagonal board with 19 tiles and 27 pieces.

The player draws a random piece from their stack and places it onto the board, repeat until full.

![Half-filled Take It Easy grid.](.github/header.png)

The final score is then calculated as follows: for every completed line, the player gets `line_length * line_value` points. The goal is to get the highest possible score.

![Score being calculated](.github/scoring.png)

The score for the above example is `121` points.

The perfect score is exactly `307`, there exist 16 such combinations.

Due to the large number of possible combinations of drawn pieces, different drawing orders and placements, the game is not fully solvable.

$$ \binom{27}{19} \cdot 19! \approx 2.7 × 10^{23} \quad \text{possible games} $$

## Usage

Install the required packages using `pip3 install -r requirements.txt`. Then either run the `python3 benchmark.py` to test the included models, or you can try out any of the provided bots interactively. Running any of these scripts should create `output.html` which shows you the bot's predictions and allows you to play the game.
 - `python3 simple.py`
 - `python3 mcts.py`
 - `python3 nn.py`

In case you want to use the `takeiteasy` library I implemented, see the following example. The implementation of `Board` under `takeiteasy/board.py` is also pretty self-documenting.

```python
from takeiteasy import Board

board = Board(seed)

while len(board.empty_tiles) > 0:
	piece = board.draw()
	tile = random.choice(board.empty_tiles)
	board.play(piece, tile)

	# Render a HTML page to visualise board
	board.show()

print(board.score())
```

If you want to use the `Maximiser` class as a starting point for your own implementation of a bot, see the following example.

```python
class MyMaximise(Maximiser):
	def heuristic(self):
		# Override this function to use the simple tactic of evaluating all possible board states for the current tile.

		# access self.board or other states
		# return a float that will be used to evaluate the state
		...

	def best_move(self, piece):
		# Override this for custom logic to choose a best move, not relying on the evaluation of all possible next states
		# return the index of the tile on which to place the piece and a dict with expectation values for all tiles (debug / visualisation aid)
		...
```

You can then test your own bot with `benchmark.py` or start a simple game with: 

```python
maximiser = MyMaximiser(Board())
maximiser.interactive()
```

## Bots

Due to the sheer number of gamestates, we have to make due with heuristics in order to approach "solving" the game.

I implemented multiple solutions:
 1. A neural-network implemented as inspired/copied from [polarbart/TakeItEasyAI](https://github.com/polarbart/TakeItEasyAI)
 2. Monte-Carlo Tree Search
 3. Simple heuristic that uses the amount of pieces that are left and the probability of drawing each piece

## Neural Network

The architecture of the neural network used to play the game was fully copied from the [polarbart/TakeItEasyAI](https://github.com/polarbart/TakeItEasyAI) repository.

I reimplemented his DQN approach to learn more about reinforcement learning.

I have included three different pretrained networks under `models/` which you can try with `python3 benchmark.py`:
 - "large.pkl" 	- hidden_size = 2048, mean=167.78, min=55, max=281
 - "medium.pkl" - hidden_size = 1024, mean=166,80, min=68, max=275
 - "mini.pkl" 	- hidden_size =  512, mean=164.69, min=43, max=268

If you want to see the models in action, go to [take-it-easy.obrhubr.org](https://take-it-easy.obrhubr.org) and let the AI play.

If you want to learn more about the models and training, check out the [detailed training logs](./models/README.md).

### Train a model yourself

#### On your own machine

Install cargo and rust then run `pip3 install -r requirements.txt` (the package `maturin` is required for the bindings).

Compile the rust bindings by running `./make.sh` (make it executable using `chmod +x make.sh` before). This should automatically install the wheel into your current environment.

You can then start training with `python3 train.py` or configure the hyperparameters first.

#### On [Google Colab](https://colab.google)

If you want to train the model on a free GPU, you can use [Google Colab](https://colab.google). Create a new notebook and paste in the following code:

```
!git clone https://github.com/obrhubr/take-it-easy
%cd take-it-easy
!pip3 install -r requirements.txt
```

Then upload the pre-built `.whl` file from the [Releases](https://github.com/obrhubr/take-it-easy/releases) and upload it to your machine. Drag it into the `take-it-easy` folder.

```
!pip3 install rust_takeiteasy-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
```

Customise the hyperparameters or simply create a new default model by running `python3 train.py`.

### Why `rust` bindings?

The `BatchedBoard` class was implemented in `rust` to speed up training massively. It allows playing `n` games at the same time. This speeds up training massively, as the bottle-neck (inference on the model) is eliminated through batch inference.

Another substantial speed-up was achieved by implementing *incremental one-hot encoding*. By only updating the encoding at the tile the new piece was placed on, you can save a lot of time compared to always starting from scratch.

## Acknowledgments

Full credit for the neural network architecture goes to [polarbart/TakeItEasyAI](https://github.com/polarbart/TakeItEasyAI).

If you have any suggestions to improve on my work or to optimise the `rust` code, I would love to hear your thoughts: find my mail on [obrhubr.org](https://obrhubr.org).