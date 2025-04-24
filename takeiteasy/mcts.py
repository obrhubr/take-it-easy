from tqdm import tqdm
import numpy as np
import random

from board import Board, N_TILES

from maximiser import Maximiser, interp

class ActionNode:
	def __init__(self, board, piece, parent=None, exploration=1):
		self.children = {}
		self.parent = parent

		self.visits = 1
		self.reward = 0

		self.board = board
		self.piece = piece
		self.exploration = exploration
		self.unfilled_tiles = [i for i in range(N_TILES) if i not in self.board.filled_tiles]
		return
	
	def is_fully_expanded(self):
		return len(self.children) >= (N_TILES - len(self.board.filled_tiles))
	
	def is_terminal(self):
		return len(self.board.filled_tiles) == N_TILES
	
	def get_action(self):
		return self.unfilled_tiles.pop(0)
	
	def best_child(self, exploration=None):
		def uct(total_reward, visits, parent_visits, c=1):
			return total_reward / visits + c * np.sqrt(np.log(parent_visits) / visits)
		
		if exploration is None:
			exploration = self.exploration

		if len(self.children) == 0:
			raise Exception("Cannot get best child of unexpanded node.")
		
		return max(
			self.children.values(), 
			key=lambda child: uct(child.reward, child.visits, self.visits, c=exploration)
		)
	
	def best_action(self):
		"""
		Pick the child with the highest visit count, then with the highest reward.
		This is maximum exploitation used after exploring the nodes to pick the best action.
		"""
		return max(self.children.values(), key=lambda c: (c.visits, c.reward))

class RandomNode:
	def __init__(self, board, tile_idx, parent=None):
		self.children = {}
		self.parent = parent

		self.visits = 1
		self.reward = 0

		self.board = board
		self.tile_idx = tile_idx
		self.unplaced_pieces = board.pieces[:]
		return
	
	def is_fully_expanded(self):
		return len(self.children) >= len(self.board.pieces)
	
	def is_terminal(self):
		return len(self.board.filled_tiles) == N_TILES
	
	def get_piece(self):
		return self.unplaced_pieces.pop()
	
	def random_outcome(self):
		return self.children[random.choice(list(self.children.keys()))]
	
	def get_child(self, piece, return_none=False):
		"""
		Get a child node by the piece placed.
		"""
		if piece in self.children:
			return self.children[piece]
		
		# Return None instead of new node
		if return_none:
			return None
		
		# If the course of action wasn't in the explored states
		return ActionNode(self.board, piece=piece)

class MCTS:
	def __init__(self, board, expansions=500, exploration=1, lookup=None, debug=False):
		self.board = board
		self.debug = debug

		# Store loaded lookup table
		self.lookup = lookup

		self.EXPANSIONS = expansions
		self.exploration = exploration

		# Cached Random node corresponding the last taken action
		self.rootnode = None
		return
	
	def simulate(self, board):
		"""
		Perform random rollouts.
		"""
		unfilled_tiles = [i for i in range(N_TILES) if i not in self.board.filled_tiles]
		random.shuffle(unfilled_tiles)

		for _ in range(N_TILES - len(board.filled_tiles)):
			piece = board.pieces.pop()
			board.play(piece, unfilled_tiles.pop(0))
		reward = board.score()

		return reward
	
	def run(self, piece: int) -> int:
		"""
		Return the best move to make with the given piece on the current board.
		Use cached MCTS from previous turns if available.
		"""
		# Use previously explored (cached) rootnode if it exists
		rootnode = ActionNode(self.board, piece=piece) if self.rootnode is None else self.rootnode.get_child(piece)

		for _ in tqdm(range(self.EXPANSIONS)) if self.debug else range(self.EXPANSIONS):
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
					tile_idx = node.get_action()

					new_board = node.board.clone()
					new_board.play(node.piece, tile_idx)
					outcome = RandomNode(board=new_board, tile_idx=tile_idx, parent=node)

					node.children[tile_idx] = outcome
					node = outcome
				else:
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

	def play_game(self) -> int:
		"""
		Play a single game and return the score.

		If a lookup table was loaded, use it.
		"""
		for _ in range(N_TILES - 1):
			piece = self.board.draw()

			# Use lookup table
			key = (piece, str(self.board.board))
			if self.lookup is not None and key in self.lookup:
				idx = self.lookup[key]
			else:
				idx, _, _ = self.run(piece)

			self.board.play(piece, idx)

		return self.board.score()

if __name__ == "__main__":
	board = Board()
	solver = MCTS(board, debug=True, expansions=100000)
	
	for _ in range(N_TILES):
		piece = solver.board.draw()
		idx, tile_values, _ = solver.run(piece)
		
		solver.board.show(tile_values=tile_values, piece=piece)

		solver.board.play(piece, idx)
		
		# Wait for confirmation
		input("Next move?")

	solver.board.show()
	print(f"Scored: {solver.board.score()}")
	