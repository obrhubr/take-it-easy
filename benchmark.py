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

def run_game(seed: int, solver_name: str) -> tuple[int, int, float]:
	"""
	Simulate a single game with the requested solver and return the results.
	"""
	board = Board(seed=seed)
	solver = select_solver(solver_name, board)

	# Run simulation
	start = time.time()
	score = solver.play_game()
	end = time.time()

	return score, board.seed, end - start

def benchmark_parallel(solver_name: str, N: int = 1000) -> tuple[list[int], list[float], list[int]]:
	"""
	Simulate N games in parallel.
	Ideal for solvers that are quick to initialise.
	Returns a list of scores, time taken to simulate and the seeds of the boards.
	"""
	scores, times, seeds = [], [], []

	rand = np.random.randint(0, N*100, size=(N)) + int(time.time())
	
	tasks = []
	with ProcessPoolExecutor() as executor:
		for num in range(N):
			tasks.append(executor.submit(run_game, int(rand[num]), solver_name))

		for future in tqdm(tasks, unit="games", desc=f"Benchmarking: {solver_name}"):
			score, seed, elapsed = future.result()
			scores.append(score)
			seeds.append(seed)
			times.append(elapsed)

	return scores, times, seeds

def benchmark(solver_name: str, N: int = 1000) -> tuple[list[int], list[float], list[int]]:
	"""
	Simulate N games one after the other, re-using the maximiser.
	Ideal for solvers with large networks or data.
	Returns a list of scores, time taken to simulate and the seeds of the boards.
	"""
	scores, times, seeds = [], [], []
	rand = np.random.randint(0, N*100, size=(N)) + int(time.time())

	solver = select_solver(solver_name, Board())
	
	for num in tqdm(range(N), desc=f"Benchmarking: {solver_name}"):
		seed = int(rand[num])
		solver.board = Board(seed=seed)
		
		start = time.time()
		score = solver.play_game()
		end = time.time()

		scores.append(score)
		seeds.append(seed)
		times.append(end - start)

	return scores, times, seeds

def analyse_output(scores: list[int], times: list[float], seeds: list[int], N: int, export_data: bool = False):
	"""
	Pretty print results of benchmark (mean, median, worst, best) and return a csv with the data.
	"""
	print()
	
	data = []
	for solver in SOLVERS:
		sc, t = scores[solver], times[solver]

		print(f"Solver {solver}:")
		print(f"Total {np.sum(sc)} after {N} iterations.")
		print(f"Took a total of {np.sum(t):.2f}s, avg of {np.mean(t):.4f} per round.")
		print(f"Mean: {np.mean(sc)}")
		print(f"Median: {np.median(sc)}")
		print(f"Worst: {np.min(sc)} - Best: {np.max(sc)}")
		print()

		if export_data:
			for score, t, seed in zip(scores[solver], times[solver], seeds[solver]):
				data += [{"solver": solver, "scores": score, "times": t, "seeds": seed }]

	if export_data:
		df = pd.DataFrame(data)
		df.to_csv("data.csv")

def run_benchmark(N: int = 1000):
	scores, times, seeds = {idx: [] for idx in SOLVERS}, {idx: [] for idx in SOLVERS}, {idx: [] for idx in SOLVERS}

	for solver_name in SOLVERS:
		if solver_name == "nn":
			scores[solver_name], times[solver_name], seeds[solver_name] = benchmark(solver_name, N)
		elif solver_name == "maximiser":
			scores[solver_name], times[solver_name], seeds[solver_name] = benchmark_parallel(solver_name, N)

	return scores, times, seeds
	
if __name__ == "__main__":
	N = 100
	scores, times, seeds = run_benchmark(N)
	analyse_output(scores, times, seeds, N)