import numpy as np

import matplotlib
import matplotlib.colors as mcolors

from board import Board, N_TILES, straights, diags_l, diags_r

class Maximiser:
	def __init__(self, board, lookup=None, debug=False, exp_coeff=0.5, real_coeff=0.5):
		self.board = board
		self.debug = debug
		self.lookup = lookup

		# Coefficients of different scores for score estimations
		self.exp_coeff = exp_coeff
		self.real_coeff = real_coeff
		return
	
	def score_probabilistic(self) -> int:
		# Count how many of the line numbers are left in the stack
		pieces_map = self.board.occurences()

		score = 0
		for rule, orientation in [
			(straights, 0),
			(diags_r, 1),
			(diags_l, 2)
		]:
			for line_indeces in rule:
				line = [self.board.board[idx] for idx in line_indeces]

				# Get the first value of the line or None
				initial = next((p[orientation] for p in line if p is not None), None)
				if initial is None:
					continue

				# Check if the evaluated line is still able to score
				if all(p[orientation] == initial if p is not None else True for p in line):
					# Calculate already filled tiles and empty ones
					filled_tiles = len(list(filter(lambda l: l is not None, line)))
					empty_tiles = len(line) - filled_tiles

					# Cumulative probability of drawing the pieces to complete that line
					n_pieces = pieces_map[initial]
					n_total = len(self.board.pieces)
					probability = np.prod([(n_pieces - n) / (n_total - n) for n in range(empty_tiles)])

					score += initial * (filled_tiles + empty_tiles * probability)
		
		return score
	
	def solve(self, piece) -> int:
		"""
		Return board with tile placed optimally and the idx at which the piece was placed.
		"""
		best_score, best_idx = -1, -1
		scores = {n: "" for n in range(N_TILES)}

		for idx in range(N_TILES):
			if idx in self.board.filled_tiles:
				continue

			self.board.board[idx] = piece
			expected_score = self.score_probabilistic()
			real_score = self.board.score()

			score = expected_score * self.exp_coeff + real_score * self.real_coeff

			if self.debug:
				scores[idx] = f"{score:.2f}"

			if score > best_score:
				best_idx = idx
				best_score = score

			self.board.board[idx] = None

		if best_idx == -1:
			raise Exception("No move found.")

		return best_idx, scores
	
	def play_game(self) -> int:
		"""
		Play a single game and return the score.
		"""
		for _ in range(N_TILES - len(self.board.filled_tiles)):
			piece = self.board.draw()

			# Use the lookup table
			key = (piece, str(self.board.board))
			if self.lookup is not None and key in self.lookup:
				idx = self.lookup[key]
			else:
				idx, _ = self.solve(piece)
				
			self.board.play(piece, idx)

		return self.board.score()
	
def interp(n, l):
	if l == 0 or l[n] is None or l[n] == "":
		return ""
	
	values = list(map(lambda n: float(n), filter(lambda n: n is not None and n != "", list(l.values()))))
	min_l, max_l = np.min(values), np.max(values)
	
	norm = mcolors.Normalize(vmin=min_l, vmax=max_l)
	cmap = matplotlib.colormaps['coolwarm']

	color = cmap(norm(float(l[n])))
	return f"{mcolors.to_hex(color)}"
	
if __name__ == "__main__":
	"""
	Step through a single game and see the predicted scores for each tile's possible placements.
	"""
	board = Board()
	solver = Maximiser(board, debug=True)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()
		idx, score_labels = solver.solve(piece)
		
		if solver.debug:
			styles = {n: f"background-color: {interp(n, score_labels)};" for n in range(N_TILES)}
		solver.board.show(label=score_labels, styles=styles, piece=piece)

		solver.board.play(piece, idx)
		# Wait for confirmation
		input("Next move?")

	print(f"Scored: {solver.board.score()}")