import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import smooth_l1_loss

from takeiteasy.board import Board, N_TILES
from takeiteasy.maximiser import Maximiser

class BatchedBoard:
	def __init__(self, batch_size: int = 1024, input_size: int = 19 * 3 * 3, output_size: int = 100):
		self.batch_size = batch_size
		self.input_size = input_size
		self.output_size = output_size

		# Init the boards
		self.reset()
		return
	
	def reset(self):
		"""
		Re-initialise all boards.
		"""
		self.boards = [Board() for _ in range(self.batch_size)]

	def scores(self) -> list[int]:
		"""
		Return a list of all boards scores.
		"""
		return [board.score() for board in self.boards]

	def states(self) -> tuple[torch.tensor, torch.tensor, torch.tensor, int]:
		"""
		Return state and (reward, next-states) for all boards.
		"""
		n_tiles = len(self.boards[0].empty_tiles)

		states = torch.zeros((self.batch_size, self.input_size), dtype=torch.float)
		next_states = torch.zeros((self.batch_size, n_tiles, self.input_size), dtype=torch.float)
		rewards = torch.zeros((self.batch_size, n_tiles), dtype=torch.float)

		for b, board in enumerate(self.boards):
			states[b] = torch.from_numpy(board.one_hot())
			
			# Enumerate each possible move and store state and reward
			piece = board.draw()
			for t, tile_idx in enumerate(board.empty_tiles):
				board.board[tile_idx] = piece
				
				# Store the board state and reward 
				next_states[b, t] = torch.from_numpy(board.one_hot())
				rewards[b, t] = torch.tensor(board.score_change(tile_idx))

				board.board[tile_idx] = None

		return states, next_states, rewards, n_tiles
	
	def play(self, tile_idxs: list[int]):
		"""
		Apply the given moves to the boards.
		"""
		for board, tile_idx in zip(self.boards, tile_idxs):
			board._play(board.empty_tiles[tile_idx])

class Network:
	"""
	Rewritten version of the network used by polarbart in https://github.com/polarbart/TakeItEasyAI.
	"""
	def __init__(self, input_size: int = N_TILES*3*3, hidden_size: int = 2048, output_size: int = 100):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.net = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.LeakyReLU(),
			nn.Linear(hidden_size, hidden_size // 2),
			nn.LeakyReLU(),
			nn.Linear(hidden_size // 2, hidden_size // 4),
			nn.LeakyReLU(),
			nn.Linear(hidden_size // 4, output_size)
		)

	def forward(self, x: torch.Tensor):
		return self.net(x)
	
	def load(self, filename = "model.pkl"):
		self.net = torch.load(filename, weights_only=False)
		return
	
	def save(self, filename = "model.pkl"):
		torch.save(self.net, filename)
		return
	
class Trainer:
	"""
	Reimplemented version of the trainer used by polarbart in https://github.com/polarbart/TakeItEasyAI.
	"""
	def __init__(
			self,
			batch_size: int = 256,
			games: int = 16384,
			game_batch_size: int = 4096,
			validation_steps: int = 8192,
			iterations: int = 100,
			epochs: int = 8,
			lr: float = 3e-4,
			lr_decay: float = 0.97,
			epsilon: float = 0.5,
			epsilon_decay: float = 0.95
		):
		if game_batch_size > validation_steps:
			raise Exception(f"Game batch size needs to be bigger than validation steps.")
		if game_batch_size > games:
			raise Exception(f"Game batch size needs to be bigger than games.")

		self.net = Network()

		# Training parameters
		self.iteration = 1
		self.iterations = iterations

		self.batch_size = batch_size
		self.games = games
		self.game_batch_size = game_batch_size
		self.validation_steps = validation_steps
		self.epochs = epochs

		# Network parameters
		self.lr = lr
		self.lr_decay = lr_decay
		self.optimizer = torch.optim.Adam(self.net.net.parameters(), self.lr)
		self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay)

		# Randomness of action taken during training set generation
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay

		# Distribution
		self.tau = (2 * torch.arange(self.net.output_size, dtype=torch.float) + 1) / (2 * self.net.output_size)

		# Logging
		self.losses = []
		self.scores = []
	
	@torch.no_grad()
	def create_dataset(self, use_net: bool = True) -> TensorDataset:
		"""
		Create the dataset for the current iteration.
		Is called dynamically as this uses the current best model to produce the data.
		
		The network is only used if use_net is True, this is to prevent nonsense results for the first iteration,
		while the net is uninitialised.
		"""
		# Training data: number_of_games * (steps_in_each_game - 1) because first step is not added to training data
		states = torch.zeros((self.games * (N_TILES - 1), self.net.input_size), dtype=torch.float)
		
		# Initialise with -1s and mask them out during loss calculation
		target_distributions = -torch.ones((self.games * ( N_TILES - 1), self.net.output_size), dtype=torch.float)
		
		# How many positions were evaluated to find the best_action
		n_samples = torch.zeros((self.games * ( N_TILES - 1),), dtype=torch.int8)

		self.net.net.eval()
		for n in tqdm(range(self.games // self.game_batch_size), desc=f"Creating dataset {self.iteration=}"):
			# Initialise boards to play all at the same time
			boards = BatchedBoard(batch_size=self.game_batch_size, input_size=self.net.input_size, output_size=self.net.output_size)

			for step in range(N_TILES):
				init_states, next_states, rewards, n_tiles = boards.states()

				# Don't use the model to predict rewards for the final piece placed
				# Here the reward is deterministic and can be fully calculated using `score_change`
				if step < N_TILES - 1 and use_net:
					qd = self.net.net(next_states.view(self.game_batch_size * n_tiles, self.net.input_size))
					qd = qd.view(self.game_batch_size, n_tiles, self.net.output_size)
				else:
					qd = torch.zeros((self.game_batch_size, n_tiles, self.net.output_size), dtype=torch.float)

				# Introduce randomness into the seen positions to prevent overfitting
				if np.random.ranf() > self.epsilon:
					# The best action is the one maximising expected score
					# Mean over dim=2 (for each move) to find mean of distribution 
					# argmax over dim=1 to find the best move for each game
					best_actions = torch.argmax(qd.mean(2), 1)
				else:
					best_actions = np.random.randint(low=0, high=n_tiles, size=(self.game_batch_size,))

				# An expected score for an empty board doesn't make sense
				# The model will only be called once a piece has been placed
				if step > 0:
					start = n * self.game_batch_size * ( N_TILES - 1) + (step - 1) * self.game_batch_size
					end = start + self.game_batch_size

					# Add (state, distribution after playing best move) to training set
					states[start:end] = init_states
					n_samples[start:end] = torch.full((self.game_batch_size,), n_tiles)
					
					# Get only the distribution for the best_actions
					indices = torch.arange(rewards.size(0))
					target_distributions[start:end] = rewards[indices, best_actions].unsqueeze(1) + qd[indices, best_actions]

				# Play best moves for all boards
				boards.play(best_actions)

			self.scores += boards.scores()
		
		return TensorDataset(states, target_distributions, n_samples)

	@torch.no_grad()
	def validate(self) -> float:
		"""
		Return the average score of the current net over `validation_steps` games.
		"""
		scores = []

		self.net.net.eval()
		for _ in tqdm(range(self.validation_steps // self.game_batch_size), "Validating"):
			boards = BatchedBoard(batch_size=self.game_batch_size, input_size=self.net.input_size, output_size=self.net.output_size)

			for step in range(N_TILES):
				_, next_states, rewards, n_tiles = boards.states()

				# Only use net before last step
				if step < N_TILES - 1:
					qd = self.net.net(next_states)
				else:
					qd = torch.zeros((self.game_batch_size, n_tiles, self.net.output_size), dtype=torch.float)
					
				best_actions = torch.argmax(qd.mean(2), 1)
				boards.play(best_actions)

			scores += boards.scores()

		self.scores += scores
		print(f"Validating model {self.iteration=}: mean={np.mean(scores):.2f}, min={np.min(scores)}, max={np.max(scores)}")
		return
	
	def quantile_regression_loss(self, qd: torch.Tensor, tqd: torch.Tensor, n_samples: int):
		"""
		Custom loss function for Distributional Quantile Regression models.
		"""
		mask = (tqd != -1)
		weight = torch.abs((self.tau - (tqd < qd.detach()).float())) / n_samples.unsqueeze(1)

		qd, tqd = torch.broadcast_tensors(qd, tqd)
		loss = (weight * mask * smooth_l1_loss(qd, tqd, reduction='none'))
		return loss.sum() / self.batch_size

	def train(self, validation_interval: int = 3):
		"""
		Train the model.
		For each iteration, generate  `self.games` games with `create_dataset`,
		then train the model on them for `self.epochs`.

		In the first iteration, don't use the net to create training data (raw rewards only)
		as the outputs will not make sense yet. This would disturb training.
		"""
		while self.iteration < self.iterations:
			dataset = self.create_dataset(use_net=self.iteration > 1)
			dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

			self.net.net.train()
			for _ in tqdm(range(self.epochs), desc=f"Training {self.iteration}"):
				for states, target_distributions, n_samples in dataloader:
					self.optimizer.zero_grad()

					qd = self.net.net(states)
					loss = self.quantile_regression_loss(qd, target_distributions, n_samples)

					loss.backward()
					self.optimizer.step()

					# Log loss
					self.losses += [loss.item()]

			# Every `validation_interval` steps, print the current average score.
			if self.iteration % validation_interval == 0:
				self.validate()

			self.lr_scheduler.step()
			self.epsilon *= self.epsilon_decay
			self.iteration += 1

			# Save model and trainer after every step
			self.save()
		return
	
	@staticmethod
	def load(filename = "trainer.pkl"):
		with open(filename, "rb") as f:
			return pickle.load(f)
	
	def save(self, filename = "trainer.pkl"):
		with open(filename, "wb") as f:
			pickle.dump(self, f)
		self.net.save()

class NNMaximiser(Maximiser):
	"""
	Implements the neural network powered maximiser. Overrides the heuristic function.
	"""
	def __init__(self, board: Board, debug: bool = False):
		# Weigh reward and neural network heuristic the same
		super().__init__(board, debug)

		self.net = Network()
		self.net.load()

	def heuristic(self) -> float:
		"""
		Use the nn to get an expected score for the current board.
		"""
		if len(self.board.filled_tiles) == N_TILES - 1:
			return 0
		
		states = torch.empty((1, N_TILES*3*3), dtype=torch.float)
		states[0] = torch.from_numpy(self.board.one_hot())
		with torch.no_grad():
			qd = self.net.net(states)

		return float(qd.mean(1)) + self.board.score()
	
if __name__ == "__main__":

	# Load the trainer from file and continue
	if False:
		trainer = Trainer.load()
	else:
		trainer = Trainer(games=256, validation_steps=256, game_batch_size=128, batch_size=128)

	trainer.train(validation_interval=1)