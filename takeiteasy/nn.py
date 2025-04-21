from board import Board, N_TILES
from lookup import interp
from maximiser import Maximiser

class NN(Maximiser):
	def heuristic(self):
		return super().heuristic()
	
if __name__ == "__main__":
	"""
	Step through a single game and see the predicted scores for each tile's possible placements.
	"""
	board = Board(seed=468100)
	solver = Maximiser(board, debug=True)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()
		idx, score_labels = solver.solve(piece)
		
		if solver.debug:
			styles = {n: f"background-color: {interp(n, score_labels)};" for n in range(N_TILES)}
		solver.board.show(label=score_labels, styles=styles, piece=piece)

		solver.board.play(piece, idx)
		# Wait for confirmation
		#input("Next move?")

	print(f"Scored: {solver.board.score()}")