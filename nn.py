import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import smooth_l1_loss

from takeiteasy.board import Board, N_TILES
from takeiteasy.maximiser import Maximiser

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
			game_batch_size: int = 1024,
			validation_steps: int = 4096,
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

		def train_batch(n: int):
			boards = [Board() for _ in range(self.game_batch_size)]

			for step in range(N_TILES):
				empty_tiles_len = N_TILES - step

				# Initialise arrays
				init_states = torch.zeros((self.game_batch_size, self.net.input_size), dtype=torch.float)
				next_states = torch.zeros((self.game_batch_size, empty_tiles_len, self.net.input_size), dtype=torch.float)
				rewards = torch.zeros((self.game_batch_size, empty_tiles_len), dtype=torch.float)
				pieces = []

				for board_idx, board in enumerate(boards):
					init_states[board_idx] = torch.from_numpy(board.one_hot())
					
					piece = board.draw()
					pieces += [piece]

					# Enumerate over each possible placement and collect the states and rewards
					for p, tile_idx in enumerate(board.empty_tiles):
						board.board[tile_idx] = piece
						
						next_states[board_idx, p] = torch.from_numpy(board.one_hot())
						rewards[board_idx, p] = torch.tensor(board.score_change(tile_idx))

						board.board[tile_idx] = None

				# Don't use the model to predict rewards for the final piece placed
				# Here the reward is deterministic and can be fully calculated using `score_change`
				if step < N_TILES - 1 and use_net:
					qd = self.net.net(next_states.view(self.game_batch_size * empty_tiles_len, self.net.input_size))
					qd = qd.view(self.game_batch_size, empty_tiles_len, self.net.output_size)
				else:
					qd = torch.zeros((self.game_batch_size, empty_tiles_len, self.net.output_size), dtype=torch.float)

				# Introduce randomness into the seen positions to prevent overfitting
				if np.random.ranf() > self.epsilon:
					# The best action is the one maximising expected score
					best_actions = torch.argmax(qd.mean(2), 1) # Mean over dimension 1 (for each of the games)
				else:
					best_actions = np.random.randint(low=0, high=len(board.empty_tiles), size=(self.game_batch_size,))

				# An expected score for an empty board doesn't make sense
				# The model will only be called once a piece has been placed
				if step > 0:
					start = n * self.game_batch_size * ( N_TILES - 1) + (step - 1) * self.game_batch_size
					end = start + self.game_batch_size

					# Add (state, distribution after playing best move) to training set
					states[start:end] = init_states
					n_samples[start:end] = torch.full((self.game_batch_size,), empty_tiles_len)
					
					# Get only the distribution for the best_actions
					indices = torch.arange(rewards.size(0))
					target_distributions[start:end] = rewards[indices, best_actions].unsqueeze(1) + qd[indices, best_actions]

				# Play the best action
				for board_idx, board in enumerate(boards):
					best_action = best_actions[board_idx]
					board.play(pieces[board_idx], board.empty_tiles[best_action])

			return [board.score() for board in boards]
		
		self.net.net.eval()
		for b in tqdm(range(self.games // self.game_batch_size), desc=f"Creating dataset {self.iteration=}"):
			self.scores += train_batch(b)
		
		return TensorDataset(states, target_distributions, n_samples)

	@torch.no_grad()
	def validate(self) -> float:
		"""
		Return the average score of the current net over `validation_steps` games.
		"""
		def validate_batch():
			scores = []
			boards = [Board() for _ in range(self.game_batch_size)]

			for step in range(N_TILES):
				all_states = []
				all_rewards = []

				all_empty_tiles = []
				all_pieces = []

				for game_idx, board in enumerate(boards):
					# Piece to be placed this round
					piece = board.draw()
					all_pieces += [piece]

					empty_tiles = board.empty_tiles
					all_empty_tiles.append(empty_tiles)

					for tile_idx in empty_tiles:
						board.board[tile_idx] = piece
						all_states += [torch.from_numpy(board.one_hot())]
						all_rewards += [board.score_change(tile_idx)]
						board.board[tile_idx] = None

				# Only use net before last step
				state_batch = torch.stack(all_states)
				if step < N_TILES - 1:
					qd_means = self.net.net(state_batch).mean(1)
				else:
					qd_means = torch.zeros((len(all_rewards)), dtype=torch.float)

				rewards_batch = torch.tensor(all_rewards)
				rewards = qd_means + rewards_batch
					
				best_indices = []
				start_idx = 0
				for empty_tiles in all_empty_tiles:
					game_rewards = rewards[start_idx:start_idx + len(empty_tiles)]

					best_idx = empty_tiles[torch.argmax(game_rewards).item()]
					best_indices.append(best_idx)

					start_idx += len(empty_tiles)
					
				for game_idx, board, piece in zip(range(self.validation_steps), boards, all_pieces):
					board.play(piece, best_indices[game_idx])

			for board in boards:
				scores.append(board.score())

			return scores

		self.net.net.eval()

		scores = []
		for _ in tqdm(range(self.validation_steps // self.game_batch_size), "Validating"):
			scores += validate_batch()

		self.scores += scores

		print(f"Validating model {self.iteration=}: mean={np.mean(scores):.2f} - min={np.min(scores)}, max={np.max(scores)}")
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
		trainer = Trainer(game_batch_size=16, games=64, validation_steps=64, batch_size=32)

	trainer.train(validation_interval=1)