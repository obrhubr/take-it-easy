from takeiteasy import Maximiser, Board
from train import Network

import torch

class NNMaximiser(Maximiser):
	"""
	Implements the neural network powered maximiser. Overrides the heuristic function.
	"""
	def __init__(self, board: Board, debug: bool = False, filename="model.pkl"):
		# Weigh reward and neural network heuristic the same
		super().__init__(board, debug)

		self.net = Network()
		self.net.load(filename=filename, device="cpu")
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
	maximiser = NNMaximiser(Board(), filename="./models/mini.pkl")
	maximiser.interactive()