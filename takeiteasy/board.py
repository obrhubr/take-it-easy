from copy import deepcopy
import numpy as np
import random
import time

PIECES = [(1, 2, 3), (1, 2, 4), (1, 2, 8), (1, 6, 3), (1, 6, 4), (1, 6, 8), (1, 7, 3), (1, 7, 4), (1, 7, 8), (5, 2, 3), (5, 2, 4), (5, 2, 8), (5, 6, 3), (5, 6, 4), (5, 6, 8), (5, 7, 3), (5, 7, 4), (5, 7, 8), (9, 2, 3), (9, 2, 4), (9, 2, 8), (9, 6, 3), (9, 6, 4), (9, 6, 8), (9, 7, 3), (9, 7, 4), (9, 7, 8)]
N_PIECES = len(PIECES)
N_TILES = 19

# Diagonals pre computed
straights = [(0, 3, 7), (1, 4, 8, 12), (2, 5, 9, 13, 16), (6, 10, 14, 17), (11, 15, 18)]
diags_r = [(0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18)]
diags_l = [(2, 6, 11), (1, 5, 10, 15), (0, 4, 9, 14, 18), (3, 8, 13, 17), (7, 12, 16)]
lines = [straights, diags_r, diags_l]

# Map from tile index on board to affeted lines
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

		# Select a playing stack
		self.pieces = deepcopy(PIECES)

		variability = 100000
		self.seed = int(time.time())*variability + random.randint(0, variability) if seed is None else seed
		random.seed(self.seed)
		random.shuffle(self.pieces)

		# Empty tiles
		self.filled_tiles = set()
		self.empty_tiles = [n for n in range(N_TILES)]
		return
	
	def clone(self):
		new_board = Board()

		# Shallow copy only
		new_board.board = self.board[:]  
		new_board.pieces = self.pieces[:]
		new_board.filled_tiles = self.filled_tiles.copy()
		# Patch for old lookup tables without the attr
		new_board.seed = self.seed if hasattr(self, "seed") else int(time.time())*1000 + random.randint(0, 1000)

		return new_board
	
	def one_hot(self):
		"""
		Return a one-hot encoded version of the current board.
		Shape: (19, 3, 3). For each tile - 3 lines, for each line - 3 possible values.

		Taken from: https://github.com/polarbart/TakeItEasyAI/blob/master/takeiteasy.py
		"""
		board = np.zeros((19, 3, 3), dtype=np.bool)
		for idx, piece in enumerate(self.board):
			if piece is not None:
				p = self.board[idx]
				board[idx, 0, (p[0] - 1) // 4] = True # straight
				board[idx, 1, 0 if p[1] == 2 else (p[1] - 5)] = True # diag_r
				board[idx, 2, 2 if p[2] == 8 else (p[2] - 3)] = 1 # diag_l
		return board.flatten()
	
	def show(self, filename="output.html", label=None, styles=None, piece=[-1, -1, -1]) -> str:
		import re
		"""
		Export board to a HTML page for easy viewing.
		"""
		def escape_style(match):
			return match.group().replace("{", "{{").replace("}", "}}")  # Double braces to escape formatting

		# Read HTML template
		with open("./takeiteasy/static_template.html", "r") as file:
			html = file.read()

		# Escape style definition braces and treat the file as a template string
		escaped_html = re.sub(r"(<style.*?>.*?</style>)", escape_style, html, flags=re.DOTALL)
		
		# Format the HTML Template
		escaped_board = list(map(lambda p: ("", "", "") if p is None else p, self.board))

		# Format tile labels
		if label is None:
			label = {n: "" for n in range(N_TILES)}
		if styles is None:
			styles = {n: "" for n in range(N_TILES)}

		board_html = escaped_html.format(
			pieces=escaped_board,
			n_tiles=len(self.filled_tiles),
			occ=self.occurences(),
			label=label,
			styles=styles,
			piece=piece,
			piece_visibility="block" if piece != [-1, -1, -1] else "none"
		)

		# write to file
		with open(filename, "w") as file:
			file.write(board_html)
	
	def show_playable(self, filename="play.html", label=None, styles=None, piece=[-1, -1, -1]) -> str:
		import re
		"""
		Export board to a HTML page that can be played on.
		"""
		# Read HTML template
		with open("./takeiteasy/play_template.html", "r") as file:
			html = file.read()

		pieces_str = str(list(map(lambda p: list(p), self.pieces)))
		board_html = re.sub(r"\{pieces_str\}", pieces_str, html)

		# Set current board state
		initial_str = str([list(map(lambda p: list(p) if p else None, self.board)), 0])
		initial_str = re.sub(r"None", "null", initial_str)
		board_html = re.sub(r"\{initial_str\}", initial_str, board_html)

		# write to file
		with open(filename, "w") as file:
			file.write(board_html)

	def occurences(self):
		piece_occ = {n: 0 for n in range(1, 10)}
		for p in self.pieces:
			if p is not None:
				piece_occ[p[0]] += 1
				piece_occ[p[1]] += 1
				piece_occ[p[2]] += 1

		return piece_occ
	
	def score_change(self, idx):
		"""
		Return the amount by which the score changed due to placing the tile at tile_idx=idx.
		"""
		score = 0

		for orientation, line_idx in enumerate(tile_to_lines[idx]):
			line = lines[orientation][line_idx]
			if all(self.board[l] for l in line) and all(self.board[l][orientation] == self.board[line[0]][orientation] for l in line):
				score += len(line) * self.board[line[0]][orientation]

		return score
	
	def score(self) -> int:
		"""
		score returns an integer which is the score of the board.

		The board (array) has the be completely filled in order for the score to be computed.
		"""
		score = 0

		for orientation, rule in enumerate(lines):
			for line_indeces in rule:
				line = [self.board[idx] for idx in line_indeces]
				
				initial = next((p[orientation] for p in line if p is not None), None)
				if all(p[orientation] == initial if p is not None else False for p in line):
					score += initial * len(line)
		return score
	
	def draw(self) -> tuple[int, int, int]:
		return self.pieces.pop(0)
	
	def play(self, piece: tuple[int, int, int], idx: int):
		self.board[idx] = piece
		self.filled_tiles.add(idx)
		self.empty_tiles.remove(idx)
		return
	
if __name__ == "__main__":
	board = Board(seed=468100)
	
	board.show()
	board.show_playable()