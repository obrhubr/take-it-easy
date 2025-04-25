from board import Board, N_TILES, LINES

class Maximiser:
	def __init__(self, board: Board, debug: bool = False, reward_coeff: float = 0, heuristic_coeff: float = 1):
		self.board = board
		self.debug = debug

		# Hyperparameters to decide scoring weights
		self.cautiousness = 4
		self.reward_coeff = reward_coeff
		self.heuristic_coeff = heuristic_coeff
	
	def heuristic(self) -> float:
		"""
		Return an expected score for the current board.
		"""
		pieces_n = self.board.occurences()

		score = 0
		for orientation, rule in enumerate(LINES):
			for line_indeces in rule:
				line = [self.board.board[idx] for idx in line_indeces]

				# Get the first value of the line or None
				initial = next((p[orientation] for p in line if p is not None), None)
				if initial is None:
					continue

				filled_tiles = list(filter(lambda l: l is not None, line))
				# Check if the evaluated line is still able to score (= all tiles are the same or empty)
				if all(p[orientation] == initial for p in filled_tiles):
					filled_n = len(filled_tiles)
					line_n = len(line)

					if line_n - filled_n > pieces_n[initial]:
						score += 0
					else:
						score += (initial * filled_n) / (line_n - filled_n + 1 + self.cautiousness)
						pieces_n[initial] -= line_n - filled_n
		
		return score
	
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
			
			score_change = self.board.score_change(idx) * self.reward_coeff if self.reward_coeff > 0 else 0
			reward = score_change + self.heuristic() * self.heuristic_coeff

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
	
if __name__ == "__main__":
	"""
	Step through a single game and see the predicted scores for each tile's possible placements.
	"""
	board = Board(seed=468100)
	solver = Maximiser(board, debug=True)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()
		idx, tile_values = solver.best_move(piece)
		
		solver.board.show(tile_values=tile_values, piece=piece)

		solver.board.play(piece, idx)
		# Wait for confirmation
		input("Next move?")

	print(f"Scored: {solver.board.score()}")