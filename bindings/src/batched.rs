use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn};
use pyo3::types::PyTuple;

use crate::board::{Board, INVALID, N_TILES};

#[pyclass]
pub struct BatchedBoard {
    size: usize,
	boards: Vec<Board>,
	pieces: Vec<u8>,
	empty_tiles: usize
}

#[pymethods]
impl BatchedBoard {
    #[new]
	fn new(size: usize) -> Self {
		let mut boards = BatchedBoard { size, boards: Vec::new(), pieces: Vec::new(), empty_tiles: N_TILES };
		
		// Initialise empty boards
		boards.reset();

		boards
	}

	fn reset(&mut self) {
		self.pieces = Vec::new();
		self.boards = Vec::new();

		for _ in 0 .. self.size {
			self.boards.push(Board::new());
		}

		self.empty_tiles = N_TILES;
	}

	fn states(&mut self, py: Python) -> Py<PyTuple> {
		let states: Vec<Vec<u8>> = self.boards.iter().map(|b| b.one_hot_rust()).collect();

		let mut next_states: Vec<Vec<Vec<u8>>> = Vec::new();
		let mut rewards: Vec<Vec<u32>> = Vec::new();

		let mut board_idx: usize = 0;
		for board in &mut self.boards {
			let piece = board.draw();

			let mut ns: Vec<Vec<u8>> = Vec::new();
			let mut r: Vec<u32> = Vec::new();
			
			// Get the next board states for all possible moves
			for empty in board.empty_tiles.clone() {
				board.board[empty as usize] = piece;

				ns.push(board.one_hot_delta(states[board_idx].clone(), piece, empty));
				r.push(board.score_change(empty));
				
				board.board[empty as usize] = INVALID;
			}

			next_states.push(ns);
			rewards.push(r);

			board_idx += 1;
		}


		let py_states = PyArray2::from_vec2(py, &states).unwrap().into_any();
		let py_next_states = PyArray3::from_vec3(py, &next_states).unwrap().into_any();
		let py_rewards = PyArray2::from_vec2(py, &rewards).unwrap().into_any();

		let results = PyTuple::new(
			py, 
			&[py_states, py_next_states, py_rewards, self.empty_tiles.into_pyobject(py).unwrap().into_any()]
		);
		results.unwrap().to_owned().into()
	}

	fn play<'py>(&mut self, moves: PyReadonlyArrayDyn<'py, u8>) {
		for (board_idx, &m) in moves.as_array().iter().enumerate() {
			let tile_idx = self.boards[board_idx].empty_tiles[m as usize];
			self.boards[board_idx].play_(tile_idx);
		}

		self.empty_tiles -= 1;
	}

	fn scores(&mut self, py: Python) -> Py<PyArray1<u32>>{
        let mut scores: Vec<u32> = Vec::new();
		
		for board in &self.boards {
			let score = board.score();
			scores.push(score)
		}

		return scores.into_pyarray(py).to_owned().into()
	}
}

#[pyfunction]
pub fn create_batched(size: usize) -> BatchedBoard {
    BatchedBoard::new(size)
}