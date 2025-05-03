import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.nn.functional import smooth_l1_loss

from takeiteasy import Maximiser, Board
from rust_takeiteasy import BatchedBoard, N_TILES, N_PIECES

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
	
	def load(self, filename = "model.pkl", device: str = "cpu"):
		self.net = torch.load(filename, weights_only=False, map_location=device)
		return
	
	def save(self, filename = "model.pkl"):
		torch.save(self.net, filename)
		return
	
class Buffer:
	"""
	Buffer class that stores the training data in uint8 and returns float on demand.
	Decreases GPU memory requirements.
	"""
	def __init__(self, n_games, input_size, output_size, device):
		# Training data: number_of_games * pieces_left * (steps_in_each_game - 1) because first step is not added to training data
		self.states = torch.zeros((n_games * (N_TILES - 1), input_size), dtype=torch.uint8, device=device)
		
		# Initialise with -1s and mask them out during loss calculation
		self.target_distributions = -torch.ones((n_games * (N_TILES - 1),  (N_PIECES - 1), output_size), dtype=torch.uint8, device=device)
		
		# How many positions were evaluated to find the best_action
		self.n_samples = torch.zeros((n_games * (N_TILES - 1),), dtype=torch.int8, device=device)

		self.device = device
		self.n_games = n_games

		self.index = 0

	def insert(self, states: torch.Tensor, td: torch.Tensor):
		batch_size = states.shape[0]
		start, end = self.index, self.index + batch_size

		assert self.index + batch_size <= self.states.shape[0], "Buffer is full."

		self.states[start:end] = states.to(self.device)
		# Only insert up to td.size(1)
		# self.target_distributions : (_, 26, _) (for all possible pieces to be placed) 
		# td size (N_PIECES - step) for dim=1 to fill
		self.target_distributions[start:end, :td.size(1)] = td.to(self.device)
		self.n_samples[start:end] = td.size(1) # Size of the n_pieces_left dimension

		self.index += batch_size
		return
	
	def __getitem__(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		return (
			self.states[idx].to(self.device).float(),
			self.target_distributions[idx].to(self.device).float(),
			self.n_samples[idx].to(self.device).long()
		)

	def __len__(self):
		return self.states.size(0)

class CustomDataLoader:
    """
    A custom dataloader which is significantly faster compared to the PyTorch dataloader if the dataset is already on the gpu
	From: https://github.com/polarbart/TakeItEasyAI/blob/master/trainer_distributional_quantile_regression.py#L47
    """
    def __init__(self, dataset: Buffer, batch_size: int, device: str, data_device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.device = device
        self.data_device = data_device
        self.batches = None
        self.idx = None

    def __iter__(self):
        self.batches = torch.randperm(len(self.dataset), device=self.data_device)[:self.num_batches * self.batch_size].view((self.num_batches, self.batch_size))
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.num_batches:
            raise StopIteration
        ret = self.dataset[self.batches[self.idx]]
        self.idx += 1
        return ret

    def __len__(self):
        return self.num_batches
	
class Trainer:
	"""
	Reimplemented version of the trainer used by polarbart in https://github.com/polarbart/TakeItEasyAI.
	"""
	def __init__(
			self,
			batch_size: int = 128,
			games: int = 16384,
			game_batch_size: int = 1024,
			validation_steps: int = 16384,
			iterations: int = 150,
			epochs: int = 8,
			lr: float = 3e-4,
			lr_decay: float = 0.97,
			epsilon: float = 0.5,
			epsilon_decay: float = 0.95,
			device: str = None,
			net_input_size: int = 19*3*3,
			net_output_size: int = 100,
			net_hidden_size: int = 2048
		):
		if game_batch_size > validation_steps:
			raise Exception(f"Game batch size needs to be bigger than validation steps.")
		if game_batch_size > games:
			raise Exception(f"Game batch size needs to be bigger than games.")

		self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
		print(f"Training using {self.device}.")

		self.net_input_size = net_input_size
		self.net_output_size = net_output_size
		self.net_hidden_size = net_hidden_size
		self.net = Network(input_size=net_input_size, output_size=net_output_size, hidden_size=net_hidden_size)
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
		self.tau = self.tau.unsqueeze(1)

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
		buffer = Buffer(self.games, self.net.input_size, self.net.output_size, self.device)

		self.net.net.eval()
		for _ in tqdm(range(self.games // self.game_batch_size), desc=f"Creating dataset {self.iteration=}"):
			# Initialise boards to play all at the same time
			boards = BatchedBoard(self.game_batch_size)

			for step in range(N_TILES):
				init_states, next_states, rewards = boards.states(step > 0) # Only try all pieces after first step
				init_states, next_states, rewards = torch.from_numpy(init_states).to(self.device), torch.from_numpy(next_states).to(self.device), torch.from_numpy(rewards).to(self.device)

				# Don't use the model to predict rewards for the final piece placed
				# Here the reward is deterministic and can be fully calculated using `score_change`
				if step < N_TILES - 1 and use_net:
					qd = self.net.net(next_states.float())
				else:
					qd = torch.zeros(
						(self.game_batch_size, N_PIECES - step, N_TILES - step, self.net.output_size),
						dtype=torch.float, 
						device=self.device
					)
				
				# Sum rewards with predicted rewards
				expected = rewards + qd.mean(3)
				# best_actions : n_games x n_pieces_left
				best_actions = expected.argmax(2) # Find the tile with the highest expected value

				# Introduce randomness into the seen positions to prevent overfitting
				actions = np.where(
					np.random.ranf(self.game_batch_size) < self.epsilon,
					np.random.randint(low=0, high=N_TILES - step, size=(self.game_batch_size,), dtype=np.uint8),

					# For each game (dim=0), get the best action given the first piece
					# This is because the first piece is the one to be actually played in the simulation
					# The other pieces are tried just to augment the training data
					best_actions[:, 0].cpu().numpy().astype(np.uint8)
				)

				# An expected score for an empty board doesn't make sense
				# The model will only be called once a piece has been placed
				if step > 0:
					# For each possible piece, get the rewards for the tiles with the highest scores
					best_rewards = rewards.gather(2, best_actions.unsqueeze(2))
					
					best_qd = qd.gather(
						2, # For each possible piece, get the distribution for the tiles with the highest score
						best_actions
							.view(self.game_batch_size, N_PIECES - step, 1, 1) # same as unsqueeze twice
							.expand(self.game_batch_size, N_PIECES - step, 1, self.net.output_size)
					).squeeze(2)

					target_distributions = best_rewards + best_qd
					buffer.insert(init_states, target_distributions)

				# Play best moves for all boards
				boards.play(actions)
		
		return buffer

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
				_, next_states, rewards = boards.states(False) # Don't return states for all possible pieces
				next_states, rewards = torch.from_numpy(next_states).to(self.device).float(), torch.from_numpy(rewards).to(self.device).float()

				# remove unnecessary dimension: n_pieces_left
				next_states, rewards = next_states.squeeze(1), rewards.squeeze(1)

				# Only use net before last step
				if step < N_TILES - 1:
					qd = self.net.net(next_states)
				else:
					qd = torch.zeros((self.game_batch_size, N_TILES - step, self.net.output_size), dtype=torch.float, device=self.device)
				
				expected = qd.mean(2) + rewards
				best_actions = torch.argmax(expected, 1)
				boards.play(best_actions.cpu().to(dtype=torch.uint8).numpy())

			scores += list(boards.scores())

		self.scores += [{ "iteration": self.iteration, "scores": scores }]
		print(f"Validating model {self.iteration=}: mean={np.mean(scores):.2f}, min={np.min(scores)}, max={np.max(scores)}")
		return
	
	def quantile_regression_loss(self, qd: torch.Tensor, tqd: torch.Tensor, n_samples: int):
		"""
		Custom loss function for Distributional Quantile Regression models.
		"""
		tqd = tqd.view(tqd.size(0), 1, -1)
		qd = qd.unsqueeze(2)

		# Divide by number of pieces remaining on stack
		# Ensures situation is weighted according to probability of drawing that specific piece
		mask = (tqd != -1)
		weight = torch.abs((self.tau - (tqd < qd.detach()).float())) / n_samples.view(-1, 1, 1)

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
			dataloader = CustomDataLoader(dataset, batch_size=self.batch_size, device=self.device, data_device=self.device)

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

			self.lr_scheduler.step()
			self.epsilon *= self.epsilon_decay

			# Every `validation_interval` steps, print the current average score.
			if self.iteration % validation_interval == 0:
				self.validate()

			# Save model and trainer after every step
			self.iteration += 1
			self.save()
		return
	
	def __getstate__(self):
		state = self.__dict__.copy()
		state['net'] = self.net.net.state_dict()
		state['optimizer'] = self.optimizer.state_dict()
		state['lr_scheduler'] = self.lr_scheduler.state_dict()
		return state

	def __setstate__(self, state):
		net = Network(input_size=state['net_input_size'], output_size=state['net_output_size'], hidden_size=state['net_hidden_size'])
		net.net.load_state_dict(state['net'])

		# Set to device if cuda is available - train using gpu if was trained on cpu originally
		state['device'] = state['device'] if torch.cuda.is_available() else "cpu"
		net.net.to(state['device'])

		optimizer = torch.optim.Adam(net.net.parameters(), state['lr'])
		lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, state['lr_decay'])

		optimizer.load_state_dict(state['optimizer'])
		lr_scheduler.load_state_dict(state['lr_scheduler'])

		state['net'] = net
		state['optimizer'] = optimizer
		state['lr_scheduler'] = lr_scheduler

		self.__dict__ = state
	
	@staticmethod
	def load(filename = "trainer.pkl", device: str = None):
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		self = torch.load(filename, weights_only=False, map_location=device)
		self.device = device
		self.net.net.to(self.device)

		return self
	
	def save(self, filename = "trainer.pkl"):
		torch.save(self, filename)
		self.net.save()

class NNMaximiser(Maximiser):
	"""
	Implements the neural network powered maximiser. Overrides the heuristic function.
	"""
	def __init__(self, board: Board, debug: bool = False):
		# Weigh reward and neural network heuristic the same
		super().__init__(board, debug)

		self.net = Network()
		self.net.load(device="cpu")
		self.net.net.eval()

	@torch.no_grad()
	def best_move(self, piece: tuple[int, int, int]) -> tuple[int, list[int]]:
		"""
		Use the nn to get an expected score for the current board.
		"""
		states = torch.zeros((len(self.board.empty_tiles), self.net.input_size), dtype=torch.float)
		rewards = torch.zeros((len(self.board.empty_tiles),), dtype=torch.float)

		tile_labels = {}
		for idx, tile in enumerate(self.board.empty_tiles):
			self.board.board[tile] = piece
			states[idx] = torch.from_numpy(self.board.one_hot()).float()
			rewards[idx] = self.board.score_change(tile)
			self.board.board[tile] = None

		if len(self.board.empty_tiles) > 1:
			qd = self.net.net(states)
			expectations = (qd + rewards.unsqueeze(1)).mean(1)
			best_action = expectations.argmax()
			
			if self.debug:
				score = self.board.score()
				tile_labels = {tile: expectations[n] + score for n, tile in enumerate(self.board.empty_tiles)}
		else:
			best_action = rewards.argmax()
			
			if self.debug:
				score = self.board.score()
				tile_labels = {tile: rewards[n] + score for n, tile in enumerate(self.board.empty_tiles)}

		return self.board.empty_tiles[best_action], tile_labels
	
if __name__ == "__main__":

	# Load the trainer from file and continue
	if False:
		trainer = Trainer.load()
	else:
		trainer = Trainer()

	trainer.train(validation_interval=1)