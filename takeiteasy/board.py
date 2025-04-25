import numpy as np
import random
import time
import re
import numpy as np

import matplotlib
import matplotlib.colors as mcolors

# List of all take-it-easy pieces
PIECES = [(1, 2, 3), (1, 2, 4), (1, 2, 8), (1, 6, 3), (1, 6, 4), (1, 6, 8), (1, 7, 3), (1, 7, 4), (1, 7, 8), (5, 2, 3), (5, 2, 4), (5, 2, 8), (5, 6, 3), (5, 6, 4), (5, 6, 8), (5, 7, 3), (5, 7, 4), (5, 7, 8), (9, 2, 3), (9, 2, 4), (9, 2, 8), (9, 6, 3), (9, 6, 4), (9, 6, 8), (9, 7, 3), (9, 7, 4), (9, 7, 8)]
N_PIECES = len(PIECES)
N_TILES = 19

# Diagonal lines pre-computed
STRAIGHT = [(0, 3, 7), (1, 4, 8, 12), (2, 5, 9, 13, 16), (6, 10, 14, 17), (11, 15, 18)]
DIAGR = [(0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18)]
DIAGL = [(2, 6, 11), (1, 5, 10, 15), (0, 4, 9, 14, 18), (3, 8, 13, 17), (7, 12, 16)]
LINES = [STRAIGHT, DIAGR, DIAGL]

# Map from tile_idx to affected lines
tile_to_lines = {
	0: [0, 0, 2],
	1: [1, 0, 1],
	2: [2, 0, 0],
	3: [0, 1, 3],
	4: [1, 1, 2],
	5: [2, 1, 1],
	6: [3, 1, 0],
	7: [0, 2, 4],
	8: [1, 2, 3],
	9: [2, 2, 2],
	10: [3, 2, 1],
	11: [4, 2, 0],
	12: [1, 3, 4],
	13: [2, 3, 3],
	14: [3, 3, 2],
	15: [4, 3, 1],
	16: [2, 4, 4],
	17: [3, 4, 3],
	18: [4, 4, 2]
}

class Board:
	def __init__(self, board=None, seed=None):
		if board is None:
			board = [None] * N_TILES
		self.board = board

		# Initialise with random seed used to shuffle tiles
		variability = 100000
		self.seed = int(time.time())*variability + random.randint(0, variability) if seed is None else seed
		random.seed(self.seed)

		# Shuffle pieces
		self.pieces = PIECES[:]
		random.shuffle(self.pieces)

		# Keep track of empty and filled tiles
		self.filled_tiles = set()
		self.empty_tiles = [n for n in range(N_TILES)]
	
	def clone(self) -> "Board":
		"""
		Return a clone of the board.
		Can safely be used instead of copying the entire object.
		"""
		new_board = Board()

		# Shallow copy only
		new_board.board = self.board[:]  
		new_board.pieces = self.pieces[:]
		new_board.filled_tiles = self.filled_tiles.copy()
		# Patch for old lookup tables without the attr
		new_board.seed = self.seed if hasattr(self, "seed") else int(time.time())*1000 + random.randint(0, 1000)

		return new_board
	
	def one_hot(self) -> np.array:
		"""
		Return a one-hot encoded version of the current board.
		Shape: (19, 3, 3) -> flattened. For each tile (3 lines), for each line (3 possible values).

		Source: https://github.com/polarbart/TakeItEasyAI/blob/master/takeiteasy.py.
		"""
		board = np.zeros((N_TILES, 3, 3), dtype=np.bool)
		for idx, piece in enumerate(self.board):
			if piece is not None:
				p = self.board[idx]
				board[idx, 0, (p[0] - 1) // 4] = True # straight
				board[idx, 1, 0 if p[1] == 2 else (p[1] - 5)] = True # diag_r
				board[idx, 2, 2 if p[2] == 8 else (p[2] - 3)] = 1 # diag_l
		return board.flatten()
	
	def show(self, filename="output.html", tile_values=None, piece=None):
		"""
		Export board to an interactive HTML page.
		If `tile_values` is specified, the unfilled tiles are labelled with their expected values.
		If `piece` is specified, the piece to be placed is shown on the page.
		"""
		def interp(n, l):
			if l == 0 or l[n] is None or l[n] == "":
				return ""
			
			values = list(map(lambda n: float(n), filter(lambda n: n is not None and n != "", list(l.values()))))
			min_l, max_l = np.min(values), np.max(values)
			
			norm = mcolors.Normalize(vmin=min_l, vmax=max_l)
			cmap = matplotlib.colormaps['coolwarm']

			color = cmap(norm(float(l[n])))
			return f"{mcolors.to_hex(color)}"
		
		# Read HTML template
		with open("./takeiteasy/takeiteasy.html", "r") as file:
			html = file.read()

		# Set remaining pieces, with current piece at the beginning
		pieces_str = str(list(map(lambda p: list(p), ([piece] if piece else []) + self.pieces)))
		board_html = re.sub(r"\{pieces_str\}", pieces_str, html)

		# Set current board state
		initial_str = str([list(map(lambda p: list(p) if p else None, self.board)), 0])
		initial_str = re.sub(r"None", "null", initial_str)
		board_html = re.sub(r"\{initial_str\}", initial_str, board_html)

		# Add tile values and styling
		if tile_values is None:
			tile_labels, styles = {n: "" for n in range(N_TILES)}, {n: "" for n in range(N_TILES)}
		else:	
			# Format tile labels and add color
			tile_labels = {n: f"{tile_values[n]:.2f}" if n in tile_values else "" for n in range(N_TILES)}
			styles = {n: f"background-color: {interp(n, tile_labels)};" for n in range(N_TILES)}

		board_html = re.sub(r"\{tile_labels\}", str(tile_labels), board_html)
		board_html = re.sub(r"\{tile_styles\}", str(styles), board_html)

		# write to file
		with open(filename, "w") as file:
			file.write(board_html)
		return

	def occurences(self) -> dict[int, int]:
		"""
		Each piece is determined uniquely by the three lines (straight, diagonal right-to-left, diagonal left-to-right).
		`occurences` returns a dictionary mapping each line number (1 to 9)
		to the number of pieces on which they appear still left in the stack.
		"""
		piece_occ = {n: 0 for n in range(1, 10)}
		for p in self.pieces:
			if p is not None:
				piece_occ[p[0]] += 1
				piece_occ[p[1]] += 1
				piece_occ[p[2]] += 1

		return piece_occ
	
	def score_change(self, idx: int) -> int:
		"""
		Return the amount by which the score changed due to placing the tile at tile_idx=idx.
		"""
		score = 0

		for orientation, line_idx in enumerate(tile_to_lines[idx]):
			line = LINES[orientation][line_idx]
			if all(self.board[l] for l in line) and all(self.board[l][orientation] == self.board[line[0]][orientation] for l in line):
				score += len(line) * self.board[line[0]][orientation]

		return score
	
	def score(self) -> int:
		"""
		`score` returns the points scored with the current board.
		The board (array) has the be completely filled in order for the score to be correctly computed.
		"""
		score = 0

		for orientation, rule in enumerate(LINES):
			for line_indeces in rule:
				line = [self.board[idx] for idx in line_indeces]
				
				initial = next((p[orientation] for p in line if p is not None), None)
				if all(p[orientation] == initial if p is not None else False for p in line):
					score += initial * len(line)
		return score
	
	def draw(self) -> tuple[int, int, int]:
		"""
		Draw a piece from the stack.
		Pops it from `board.pieces`.
		"""
		return self.pieces.pop(0)
	
	def play(self, piece: tuple[int, int, int], idx: int):
		"""
		Place a piece on the board, at the specified index.
		Removes the idx from the empty tiles and adds it to the filled tiles.
		"""
		if idx < 0:
			raise Exception(f"Illegal Move: Cannot place piece at {idx=}.")
		
		self.board[idx] = piece
		self.filled_tiles.add(idx)
		self.empty_tiles.remove(idx)
	
if __name__ == "__main__":
	board = Board(seed=468100)
	board.show()