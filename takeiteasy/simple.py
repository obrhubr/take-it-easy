from board import Board, N_TILES, LINES
from maximiser import Maximiser

class SimpleMaximiser(Maximiser):
	def __init__(self, board: Board, debug: bool = False, cautiousness: int = 4):
		super().__init__(board, debug)

		# Heuristic parameter
		self.cautiousness = cautiousness
	
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
	
if __name__ == "__main__":
	maximiser = SimpleMaximiser(Board())
	maximiser.interactive()