from tqdm import tqdm
import pickle as pkl
import os

from board import Board, N_TILES
from mcts import MCTS, RandomNode

def create_lookup(lookup=None, expansions=1000, exploration=10, debug=False):
	print(f"Creating lookup table with expansions={expansions} and explorations={exploration} .")

	board = Board()
	rootnode = RandomNode(board.clone(), tile_idx=-1) if lookup is None else lookup
	rootnode.visits = expansions

	# Catch early aborts and exit gracefully
	try:
		for piece in tqdm(board.pieces):
			child = rootnode.get_child(piece, return_none=True)

			# If this child has already been expanded, skip it
			if child is not None and child.visits >= expansions:
				print(f"Skip placing {piece} because the action was already explored up to depth {expansions}")
				continue

			# Create MCTS solver with high number of iterations and maximum expansion
			board = Board()

			# Expand only to the limit set
			exp = expansions if child is None else expansions - child.visits
			solver = MCTS(board, expansions=exp, exploration=exploration, debug=debug)

			# If the ActionNode for that piece already exists, use it
			if child is not None:
				solver.rootnode = rootnode

			_, _, actionnode = solver.run(piece)

			if child is None:
				rootnode.children[piece] = actionnode
	except KeyboardInterrupt:
		print("End training early.")
		return rootnode

	return rootnode

def get_best_moves(lookup, threshold=1000):
	best_moves = {}

	def gbm(root):		
		if type(root) is RandomNode:
			for piece in root.board.pieces:
				if piece in root.children:
					gbm(root.children[piece])
		else:
			try:
				best_child = root.best_action()
				best_moves[(root.piece, str(root.board.board))] = best_child.tile_idx
			except:
				return

			if not root.visits < threshold:
				return gbm(best_child)

	gbm(lookup)
	return best_moves

def export_lookup(lookup, filename="lookup.pkl", export_best_moves=True, best_moves_filename="best_moves.pkl", best_move_treshold=10000):
	with open(filename, "wb") as f:
		pkl.dump(lookup, f)

	if export_best_moves:
		with open(best_moves_filename, "wb") as f:
			pkl.dump(get_best_moves(lookup, best_move_treshold), f)

	print(f"Exported lookup table with {len(lookup.children)} children.")
	print(f"The exported table was generated with {lookup.visits} expansions.")
	print(f"Exported lookup table {os.path.getsize(filename) / 1024 / 1024:.2f} MB.")
	print()
	return

def load_lookup(filename="lookup.pkl", load_none=False):
	if not os.path.exists(filename):
		print("File does not exist.")
		if load_none:
			return None
		else:
			raise FileNotFoundError()

	with open(filename, "rb") as f:
		lookup = pkl.load(f)
	
	print(f"Loaded lookup table with {len(lookup.children)} children.")
	print(f"The loaded table was generated with {lookup.visits} expansions.")
	print()
	return lookup

def load_best_moves(filename="best_moves.pkl"):
	with open(filename, "rb") as f:
		return pkl.load(f)

def train(expansions=100, exploration=10, best_move_treshold=1000):
	# If it exists, load lookup
	lookup = load_lookup(load_none=True)
	
	try:
		lookup = create_lookup(lookup=lookup, expansions=expansions, exploration=exploration, debug=True)
		export_lookup(lookup, best_move_treshold=best_move_treshold)
	except KeyboardInterrupt:
		print("End training early.")
		export_lookup(lookup, best_move_treshold=best_move_treshold)

def play(expansions=100, exploration=1, seed=None, filename="./best_moves.pkl"):
	lookup = load_best_moves(filename=filename)

	# Instantiate solver
	board = Board(seed=seed)
	solver = MCTS(board, expansions=expansions, exploration=exploration, debug=True)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()

		# If the lookup table ends, reset hyperparams to normal
		if (piece, str(board.board)) in lookup:
			idx = lookup[(piece, str(board.board))]
			print(f"Used lookup table to determine best move - {idx}.")
			tile_values = {}
		else:
			idx, tile_values, _ = solver.run(piece)
		
		if solver.debug:
			solver.board.show(tile_values=tile_values, piece=piece)

		solver.board.play(piece, idx)

		# Wait for confirmation
		input("Next move?")

	solver.board.show()
	print(f"Scored: {solver.board.score()}")

if __name__ == "__main__":
	if True:
		train(expansions=300000, exploration=20, best_move_treshold=150000)
	
	if True:
		play(expansions=40000, exploration=0.1, seed=1745173593146)