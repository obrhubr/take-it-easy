import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import smooth_l1_loss

from takeiteasy import Maximiser, Board
from rust_takeiteasy import BatchedBoard, N_TILES

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
			epsilon_decay: float = 0.97,
			device: str = None
		):
		if game_batch_size > validation_steps:
			raise Exception(f"Game batch size needs to be bigger than validation steps.")
		if game_batch_size > games:
			raise Exception(f"Game batch size needs to be bigger than games.")

		self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
		print(f"Training using {self.device}.")

		self.net = Network()
		self.net.net.to(self.device)

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
		self.tau = (2 * torch.arange(self.net.output_size, dtype=torch.float, device=self.device) + 1) / (2 * self.net.output_size)

		# Logging
		self.epsilons = []
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
		states = torch.zeros((self.games * (N_TILES - 1), self.net.input_size), dtype=torch.float, device=self.device)
		
		# Initialise with -1s and mask them out during loss calculation
		target_distributions = -torch.ones((self.games * ( N_TILES - 1), self.net.output_size), dtype=torch.float, device=self.device)
		
		# How many positions were evaluated to find the best_action
		n_samples = torch.zeros((self.games * ( N_TILES - 1),), dtype=torch.int8)

		self.net.net.eval()
		for n in tqdm(range(self.games // self.game_batch_size), desc=f"Creating dataset {self.iteration=}"):
			# Initialise boards to play all at the same time
			boards = BatchedBoard(self.game_batch_size)

			for step in range(N_TILES):
				init_states, next_states, rewards, n_tiles = boards.states()
				init_states, next_states, rewards = torch.from_numpy(init_states).to(self.device).float(), torch.from_numpy(next_states).to(self.device).float(), torch.from_numpy(rewards).to(self.device).float()

				# Don't use the model to predict rewards for the final piece placed
				# Here the reward is deterministic and can be fully calculated using `score_change`
				if step < N_TILES - 1 and use_net:
					qd = self.net.net(next_states.view(self.game_batch_size * n_tiles, self.net.input_size))
					qd = qd.view(self.game_batch_size, n_tiles, self.net.output_size)
				else:
					qd = torch.zeros((self.game_batch_size, n_tiles, self.net.output_size), dtype=torch.float, device=self.device)

				# Introduce randomness into the seen positions to prevent overfitting
				if np.random.ranf() > self.epsilon:
					# The best action is the one maximising expected score
					# Mean over dim=2 (for each move) to find mean of distribution 
					# argmax over dim=1 to find the best move for each game
					best_actions = torch.argmax(qd.mean(2), 1)
				else:
					best_actions = torch.from_numpy(np.random.randint(low=0, high=n_tiles, size=(self.game_batch_size,)))

				# An expected score for an empty board doesn't make sense
				# The model will only be called once a piece has been placed
				if step > 0:
					start = n * self.game_batch_size * ( N_TILES - 1) + (step - 1) * self.game_batch_size
					end = start + self.game_batch_size

					# Add (state, distribution after playing best move) to training set
					states[start:end] = init_states
					n_samples[start:end] = torch.full((self.game_batch_size,), n_tiles, device=self.device)
					
					# Get only the distribution for the best_actions
					indices = torch.arange(rewards.size(0))
					target_distributions[start:end] = rewards[indices, best_actions].unsqueeze(1) + qd[indices, best_actions]

				# Play best moves for all boards
				boards.play(best_actions.to(dtype=torch.uint8).cpu().numpy())
		
		return TensorDataset(states, target_distributions, n_samples)

	@torch.no_grad()
	def validate(self) -> float:
		"""
		Return the average score of the current net over `validation_steps` games.
		"""
		scores = []

		self.net.net.eval()
		for _ in tqdm(range(self.validation_steps // self.game_batch_size), "Validating"):
			boards = BatchedBoard(self.game_batch_size)

			for step in range(N_TILES):
				_, next_states, rewards, n_tiles = boards.states()
				next_states, rewards = torch.from_numpy(next_states).to(self.device).float(), torch.from_numpy(rewards).to(self.device).float()

				# Only use net before last step
				if step < N_TILES - 1:
					qd = self.net.net(next_states)
				else:
					qd = torch.zeros((self.game_batch_size, n_tiles, self.net.output_size), dtype=torch.float)
				
				expected = qd.mean(2) + rewards
				best_actions = torch.argmax(expected, 1)
				boards.play(best_actions.to(dtype=torch.uint8).numpy())

			scores += scores

		self.scores += [{ "iteration": self.iteration, "scores": list(boards.scores()) }]
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
			self.epsilons += [{ "iteration": self.iteration, "epsilon": self.epsilon }]

			dataset = self.create_dataset(use_net=self.iteration > 1)
			dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

			self.net.net.train()
			losses = []
			for _ in tqdm(range(self.epochs), desc=f"Training {self.iteration}"):
				for states, target_distributions, n_samples in dataloader:
					self.optimizer.zero_grad()

					qd = self.net.net(states)
					loss = self.quantile_regression_loss(qd, target_distributions, n_samples)

					loss.backward()
					self.optimizer.step()

					# Log loss
					losses += [loss.item()]
				
			self.losses += [{ "iteration": self.iteration, "loss": losses }]

			# Every `validation_interval` steps, print the current average score.
			if self.iteration % validation_interval == 0:
				self.validate()

			self.lr_scheduler.step()
			self.epsilon *= self.epsilon_decay
			self.iteration += 1

			# Save model and trainer after every step
			self.save()
		return
	
	def __getstate__(self):
		state = self.__dict__.copy()
		state['net'] = self.net
		state['optimizer'] = self.optimizer.state_dict()
		state['lr_scheduler'] = self.lr_scheduler.state_dict()
		return state

	def __setstate__(self, state):
		net = state['net']
		optimizer = torch.optim.Adam(net.net.parameters(), state['lr'])
		lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, state['lr_decay'])

		optimizer.load_state_dict(state['optimizer'])
		lr_scheduler.load_state_dict(state['lr_scheduler'])

		state['optimizer'] = optimizer
		state['lr_scheduler'] = lr_scheduler

		self.__dict__ = state
	
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
		self.net.net.eval()

	def best_move(self, piece: tuple[int, int, int]) -> tuple[int, list[int]]:
		"""
		Use the nn to get an expected score for the current board.
		"""
		states = torch.zeros((len(self.board.empty_tiles), self.net.input_size), dtype=torch.float)
		rewards = torch.zeros((len(self.board.empty_tiles),), dtype=torch.float)

		for idx, tile in enumerate(self.board.empty_tiles):
			self.board.board[tile] = piece
			states[idx] = torch.from_numpy(self.board.one_hot()).float()
			rewards[idx] = self.board.score_change(tile)
			self.board.board[tile] = None

		if len(self.board.empty_tiles) > 1:
			qd = self.net.net(states)
			best_action = (qd + rewards.unsqueeze(1)).mean(1).argmax()
		else:
			best_action = rewards.argmax()

		return self.board.empty_tiles[best_action], {}
	
if __name__ == "__main__":

	# Load the trainer from file and continue
	if False:
		trainer = Trainer.load()
	else:
		trainer = Trainer()

	trainer.train(validation_interval=1)