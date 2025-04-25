from tqdm import tqdm
import numpy as np
import random

from takeiteasy import Board, N_TILES

class ActionNode:
	"""
	ActionNodes represent tile placement steps in the MCTS graph.
	At this node, the predetermined piece is placed onto the board for every unfilled tile.
	"""
	def __init__(self, board: Board, piece: tuple[int, int, int], parent: "RandomNode" = None, exploration: float = 1):
		self.children = {}
		self.parent = parent

		self.visits = 1
		self.reward = 0

		self.board = board
		self.piece = piece
		self.exploration = exploration

		# Keep track of which tiles were not yet explored
		self.unfilled_tiles = [i for i in range(N_TILES) if i not in self.board.filled_tiles]
	
	def is_fully_expanded(self) -> bool:
		"""
		Return if a child node was generated for each open tile.
		For the first ActionNode, this would be 19 child RandomNodes.
		"""
		return len(self.children) >= (N_TILES - len(self.board.filled_tiles))
	
	def is_terminal(self) -> bool:
		"""
		Return True if the board is filled.
		"""
		return len(self.board.filled_tiles) == N_TILES
	
	def get_action(self) -> int:
		"""
		Get first unfilled tile_idx from open tiles and remove from queue.
		"""
		return self.unfilled_tiles.pop(0)
	
	def best_child(self, exploration: float | None = None) -> "RandomNode":
		"""
		Return the best child RandomNode according to UCT metric.
		Maximise balance between exploration and exploitation.
		"""
		def uct(total_reward, visits, parent_visits, c=1):
			return total_reward / visits + c * np.sqrt(np.log(parent_visits) / visits)
		
		if exploration is None:
			exploration = self.exploration

		if len(self.children) == 0:
			raise Exception("Cannot get best child of unexpanded node.")
		
		# Custom key function for max
		return max(
			self.children.values(), 
			key=lambda child: uct(child.reward, child.visits, self.visits, c=exploration)
		)
	
	def best_action(self) -> "RandomNode":
		"""
		Pick the child with the highest visit count, then with the highest reward.
		This function should be called after exploring the nodes to pick the best action.
		"""
		return max(self.children.values(), key=lambda c: (c.visits, c.reward))

class RandomNode:
	"""
	RandomNodes represent the step of drawing a random piece from the stack.
	"""
	def __init__(self, board: Board, tile_idx: int, parent: ActionNode = None):
		self.children = {}
		self.parent = parent

		self.visits = 1
		self.reward = 0

		self.board = board
		self.tile_idx = tile_idx
		self.unplaced_pieces = board.pieces[:]
	
	def is_fully_expanded(self) -> bool:
		"""
		Return True, if a childnode was created for all possible pieces left on the stack.
		"""
		return len(self.children) >= len(self.board.pieces)
	
	def is_terminal(self) -> bool:
		"""
		Return true if all tiles were filled.
		"""
		return len(self.board.filled_tiles) == N_TILES
	
	def get_piece(self) -> tuple[int, int, int]:
		"""
		Return an unplaced piece and remove from stack.
		"""
		return self.unplaced_pieces.pop()
	
	def random_outcome(self) -> ActionNode:
		"""
		Return a random child, to simulate random piece picking process.
		"""
		return self.children[random.choice(list(self.children.keys()))]
	
	def get_child(self, piece: tuple[int, int, int]) -> ActionNode:
		"""
		Get a child node by the piece placed.
		"""
		if piece in self.children:
			return self.children[piece]
		
		# If the course of action wasn't in the explored states
		return ActionNode(self.board, piece=piece)

class MCTSMaximiser:
	"""
	Monte-Carlo-Tree-Search implemented for take-it-easy.
	"""
	def __init__(self, board: Board, expansions: int = 500, exploration: float = 1, debug: bool = False):
		self.board = board
		self.debug = debug

		# Parameters
		self.expansions = expansions
		self.exploration = exploration

		# Cached Random node corresponding the last taken action
		self.rootnode = None
		return
	
	def simulate(self, board) -> int:
		"""
		Perform random rollouts and return the score
		"""
		unfilled_tiles = [i for i in range(N_TILES) if i not in self.board.filled_tiles]
		random.shuffle(unfilled_tiles)

		for _ in range(N_TILES - len(board.filled_tiles)):
			piece = board.pieces.pop()
			board.play(piece, unfilled_tiles.pop(0))

		return board.score()
	
	def best_move(self, piece: tuple[int, int, int]) -> tuple[int, dict[int, float], ActionNode | RandomNode]:
		"""
		Return the best move to make with the given piece on the current board.
		Uses cached MCTS graph from previous turns.

		Overrides the default maximiser `best_move` function.
		"""
		# Use previously explored (cached) rootnode if it exists
		rootnode = ActionNode(self.board, piece=piece) if self.rootnode is None else self.rootnode.get_child(piece)

		for _ in tqdm(range(self.expansions)) if self.debug else range(self.expansions):
			node = rootnode

			# Selection
			while node.is_fully_expanded() and not node.is_terminal():
				if type(node) is ActionNode:
					node = node.best_child()
				else:
					node = node.random_outcome()

			# Expansion
			if not node.is_terminal():
				if type(node) is ActionNode:
					# Action node: set piece on not yet explored tile
					tile_idx = node.get_action()

					new_board = node.board.clone()
					new_board.play(node.piece, tile_idx)
					outcome = RandomNode(board=new_board, tile_idx=tile_idx, parent=node)

					node.children[tile_idx] = outcome
					node = outcome
				else:
					# Random node: Get random piece from pieces left that were not yet simulated
					piece = node.get_piece()
					outcome = ActionNode(node.board.clone(), piece, parent=node, exploration=self.exploration)
					node.children[piece] = outcome

					node = outcome

			# Simulation
			reward = self.simulate(node.board.clone())
			
			# Backpropagation
			while node is not None:
				node.reward = (node.reward * node.visits + reward) / (node.visits + 1)
				node.visits += 1
				node = node.parent

		# Return the scores of each possible tile idx for visualisation purposes
		tile_values = {}
		if self.debug:
			for c in rootnode.children.values():
				tile_values[c.tile_idx] = c.reward

		# Get the best action and set the new cached rootnode
		best_action = rootnode.best_action()
		self.rootnode = best_action

		return best_action.tile_idx, tile_values, rootnode

if __name__ == "__main__":
	board = Board()
	solver = MCTSMaximiser(board, debug=True, expansions=100000)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()
		idx, tile_values, _ = solver.best_move(piece)

		solver.board.show(tile_values=tile_values, piece=piece)
		solver.board.play(piece, idx)
		
		# Wait for confirmation
		input("Next move?")

	solver.board.show()
	print(f"Scored: {solver.board.score()}")
	