from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from board import Board

from maximiser import Maximiser
from nn import NNMaximiser
from lookup import load_best_moves

SOLVERS = ["nn", "maximiser" "lookup"]

# Load preprocessed lookup table only if it's needed
if "lookup" in SOLVERS:
	lookup = load_best_moves()

def select_solver(solver_name, board):
	if solver_name == "maximiser":
		solver = Maximiser(deepcopy(board))
	elif solver_name == "nn":
		solver = NNMaximiser(deepcopy(board))
	elif solver_name == "lookup":
		solver = Maximiser(deepcopy(board), lookup=lookup)
	else:
		raise Exception(f"Solver \"{solver_name}\" does not exist.")
	
	return solver

def run_game(seed, solver_idx, solver_name):
	board = Board(seed=seed)

	solver = select_solver(solver_name, board)

	# Run simulation
	start = time.time()
	score = solver.play_game()
	end = time.time()

	return solver_idx, score, board.seed, end - start

def benchmark_parallel(N=1000):
	scores, times, seeds = {idx: [] for idx in range(len(SOLVERS))}, {idx: [] for idx in range(len(SOLVERS))}, {idx: [] for idx in range(len(SOLVERS))}

	rand = np.random.randint(0, N*100, size=(N)) + int(time.time())
	
	tasks = []
	with ProcessPoolExecutor() as executor:
		for num in range(N):
			for idx, solver_name in enumerate(SOLVERS):
				tasks.append(executor.submit(run_game, int(rand[num]), idx, solver_name))

		for future in tqdm(tasks, unit="games"):
			idx, score, seed, elapsed = future.result()
			scores[idx].append(score)
			seeds[idx].append(seed)
			times[idx].append(elapsed)

	return scores, times, seeds

def analyse_output(scores, times, seeds, N):
	data = []
	for idx, solver in enumerate(SOLVERS):
		sc, t = scores[idx], times[idx]

		print(f"Solver {solver}:")
		print(f"Total {np.sum(sc)} after {N} iterations.")
		print(f"Took a total of {np.sum(t):.2f}s, avg of {np.mean(t):.4f} per round.")
		print(f"Mean: {np.mean(sc)}")
		print(f"Median: {np.median(sc)}")
		print(f"Worst: {np.min(sc)} - Best: {np.max(sc)}")
		print()

		for score, t, seed in zip(scores[idx], times[idx], seeds[idx]):
			data += [{"solver": solver, "scores": score, "times": t, "seeds": seed }]

	df = pd.DataFrame(data)
	df.to_csv("data.csv")
	
if __name__ == "__main__":
	N = 5000
	scores, times, seeds = benchmark_parallel(N)
	analyse_output(scores, times, seeds, N)