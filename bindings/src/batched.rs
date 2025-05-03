use pyo3::prelude::*;
use ndarray::Array4;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArrayDyn};
use pyo3::types::PyTuple;

use crate::board::{Board, INVALID, N_TILES};

#[pyclass]
pub struct BatchedBoard {
    size: usize,
	boards: Vec<Board>,
	pieces: Vec<u8>
}

#[pymethods]
impl BatchedBoard {
    #[new]
	fn new(size: usize) -> Self {
		let mut boards = BatchedBoard { size, boards: Vec::new(), pieces: Vec::new() };
		
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
	}

	fn states(&mut self, py: Python, iter_over_pieces: bool) -> Py<PyTuple> {
		// n_games x one_hot_encoded_size
		let states: Vec<Vec<u8>> = self.boards.iter().map(|b| b.one_hot_rust()).collect();

		// n_games x n_pieces_left (27 - step) x n_tiles_left (19 - step) x one_hot_encoded_size
		let mut next_states: Vec<Vec<Vec<Vec<u8>>>> = Vec::new();
		// n_games x n_pieces_left (27 - step) x n_tiles_left (19 - step)
		let mut rewards: Vec<Vec<Vec<i64>>> = Vec::new();

		let mut board_idx: usize = 0;
		for board in &mut self.boards {
			let mut ns_board: Vec<Vec<Vec<u8>>> = Vec::new();
			let mut r_board: Vec<Vec<i64>> = Vec::new();

			// Go throug the pieces left on the stack
			// Start on the back, due to board.draw() taking the last piece
			// And iterate over all possibilities
			for piece in board.pieces.iter().rev() {
				let mut ns_piece: Vec<Vec<u8>> = Vec::new();
				let mut r_piece: Vec<i64> = Vec::new();

				// Get the next board states for all possible moves
				for empty in board.empty_tiles.clone() {
					board.board[empty as usize] = *piece;

					ns_piece.push(board.one_hot_delta(states[board_idx].clone(), *piece, empty));
					r_piece.push(board.score_change(empty));
					
					board.board[empty as usize] = INVALID;
				}

				ns_board.push(ns_piece);
				r_board.push(r_piece);

				if !iter_over_pieces {
					break;
				}
			}

			next_states.push(ns_board);
			rewards.push(r_board);

			board_idx += 1;

			// Draw a piece and set it as the one to be placed
			board.draw();
		}


		let py_states = PyArray2::from_vec2(py, &states).unwrap().into_any();
		let py_rewards = PyArray3::from_vec3(py, &rewards).unwrap().into_any();

		// Convert next_states to PyArray4
		let d1 = next_states.len();
		let d2 = next_states[0].len();
		let d3 = next_states[0][0].len();
		let d4 = next_states[0][0][0].len();

		// Flatten into contiguous buffer
		let mut buffer: Vec<u8> = Vec::with_capacity(d1 * d2 * d3 * d4);
		for i in 0..d1 {
			for j in 0..d2 {
				for k in 0..d3 {
					buffer.extend(&next_states[i][j][k]);
				}
			}
		}
		let next_states_array = Array4::from_shape_vec((d1, d2, d3, d4), buffer).unwrap();
		let py_next_states = PyArray4::from_owned_array(py, next_states_array).into_any();

		let results = PyTuple::new(
			py, 
			&[
				py_states,
				py_next_states,
				py_rewards
			]
		);
		results.unwrap().to_owned().into()
	}

	fn play<'py>(&mut self, moves: PyReadonlyArrayDyn<'py, u8>) {
		for (board_idx, &m) in moves.as_array().iter().enumerate() {
			let tile_idx = self.boards[board_idx].empty_tiles[m as usize];
			self.boards[board_idx].play_(tile_idx);
		}
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