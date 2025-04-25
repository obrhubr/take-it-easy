from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from takeiteasy import Board

from simple import SimpleMaximiser
from nn import NNMaximiser

# Specify the solvers to benchmark
SOLVERS = ["nn", "maximiser"]

def select_solver(solver_name: str, board: Board) -> SimpleMaximiser | NNMaximiser:
	"""
	Return the requested solver instantiated with the board.
	"""
	if solver_name == "maximiser":
		solver = SimpleMaximiser(board.clone())
	elif solver_name == "nn":
		solver = NNMaximiser(board.clone())
	else:
		raise Exception(f"Solver \"{solver_name}\" does not exist.")
	
	return solver

def run_game(seed: int, solver_idx: int, solver_name: str) -> tuple[int, int, int, float]:
	"""
	Simulate a single game with the requested solver and return the results.
	"""
	board = Board(seed=seed)

	solver = select_solver(solver_name, board)

	# Run simulation
	start = time.time()
	score = solver.play_game()
	end = time.time()

	return solver_idx, score, board.seed, end - start

def benchmark_parallel(N: int = 1000) -> tuple[list[int], list[float], list[int]]:
	"""
	Simulate N games in parallel to benchmark the different solvers.
	Returns a list of scores, time taken to simulate and the seeds of the boards.
	"""
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

def analyse_output(scores: list[int], times: list[float], seeds: list[int], N: int, export_data: bool = False):
	"""
	Pretty print results of benchmark (mean, median, worst, best) and return a csv with the data.
	"""
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

		if export_data:
			for score, t, seed in zip(scores[idx], times[idx], seeds[idx]):
				data += [{"solver": solver, "scores": score, "times": t, "seeds": seed }]

	if export_data:
		df = pd.DataFrame(data)
		df.to_csv("data.csv")
	
if __name__ == "__main__":
	N = 100
	scores, times, seeds = benchmark_parallel(N)
	analyse_output(scores, times, seeds, N)