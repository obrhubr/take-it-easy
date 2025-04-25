from .board import Board, N_TILES

class Maximiser:
	def __init__(self, board: Board, rcoeff: float = 1, hcoeff: float = 1, debug: bool = False):
		self.board = board
		self.debug = debug

		# Hyperparameters
		self.rcoeff = rcoeff
		self.hcoeff = hcoeff
	
	def heuristic(self) -> float:
		return 0 # Not implemented for base class
	
	def best_move(self, piece: tuple[int, int, int]) -> tuple[int, list[int]]:
		"""
		Return the tile_idx of the best move and the expected rewards for every tile.
		Uses the reward (score change due to placing tile) and heuristic to determine the best move.
		"""
		best_reward, best_idx = -1, -1
		rewards = {}

		for idx in range(N_TILES):
			if idx in self.board.filled_tiles:
				continue

			self.board.board[idx] = piece
			
			reward = self.board.score_change(idx) * self.rcoeff + self.heuristic() * self.hcoeff

			if self.debug:
				rewards[idx] = reward

			if reward > best_reward:
				best_idx = idx
				best_reward = reward

			self.board.board[idx] = None

		if best_idx == -1:
			raise Exception("No move found.")

		return best_idx, rewards
	
	def play_game(self) -> int:
		"""
		Play a single game and return the score.
		Determines the best move at every turn using `Maximiser.best_move`.
		"""
		for _ in range(N_TILES - len(self.board.filled_tiles)):
			piece = self.board.draw()
			idx, _ = self.best_move(piece)
			self.board.play(piece, idx)

		return self.board.score()
	
	def interactive(self, wait: bool = True, seed: int | None = None):
		"""
		Step through a single game and see the predicted scores for each tile's possible placements.
		Wait for confirmation before next turn if `wait` is True.
		"""
		self.board = Board(seed)
		self.debug = True
		
		for _ in range(N_TILES):
			piece = self.board.draw()
			idx, tile_values = self.best_move(piece)
			
			self.board.show(tile_values=tile_values, piece=piece)

			self.board.play(piece, idx)

			if wait:
				# Wait for confirmation
				input("Next move?")

		self.board.show()

		print(f"Scored: {self.board.score()}")