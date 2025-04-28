use std::usize;
use rand::{seq::SliceRandom, rng};
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

pub static PIECES: [[u8; 3]; 27] = [[1, 2, 3], [1, 2, 4], [1, 2, 8], [1, 6, 3], [1, 6, 4], [1, 6, 8], [1, 7, 3], [1, 7, 4], [1, 7, 8], [5, 2, 3], [5, 2, 4], [5, 2, 8], [5, 6, 3], [5, 6, 4], [5, 6, 8], [5, 7, 3], [5, 7, 4], [5, 7, 8], [9, 2, 3], [9, 2, 4], [9, 2, 8], [9, 6, 3], [9, 6, 4], [9, 6, 8], [9, 7, 3], [9, 7, 4], [9, 7, 8]];
pub static N_PIECES: usize = PIECES.len();
pub static N_TILES: usize = 19;

pub static INVALID: u8 = 255;

#[pyclass]
pub struct Board {
    pub board: [u8; N_TILES],
    pub pieces: Vec<u8>,
    pub empty_tiles: Vec<u8>,
    pub piece: u8
}

#[pymethods]
impl Board {
    // Lookup table from tile index to LINES
    const TILE_LINES: [[u8; 3]; N_TILES] = [
        [0, 0, 2],
        [1, 0, 1],
        [2, 0, 0],
        [0, 1, 3],
        [1, 1, 2],
        [2, 1, 1],
        [3, 1, 0],
        [0, 2, 4],
        [1, 2, 3],
        [2, 2, 2],
        [3, 2, 1],
        [4, 2, 0],
        [1, 3, 4],
        [2, 3, 3],
        [3, 3, 2],
        [4, 3, 1],
        [2, 4, 4],
        [3, 4, 3],
        [4, 4, 2]
    ];

    // Precomputed tile_idx in straight lines, diagonal right-to-left, diagonal left-to-right
    const LINES: [[[u8; 5]; 5]; 3] = [
        [[0, 3, 7, INVALID, INVALID], [1, 4, 8, 12, INVALID], [2, 5, 9, 13, 16], [6, 10, 14, 17, INVALID], [11, 15, 18, INVALID, INVALID]],
        [[0, 1, 2, INVALID, INVALID], [3, 4, 5, 6, INVALID], [7, 8, 9, 10, 11], [12, 13, 14, 15, INVALID], [16, 17, 18, INVALID, INVALID]],
        [[2, 6, 11, INVALID, INVALID], [1, 5, 10, 15, INVALID], [0, 4, 9, 14, 18], [3, 8, 13, 17, INVALID], [7, 12, 16, INVALID, INVALID]]
    ];

    #[new]
    pub fn new() -> Self {
        let mut rng = rng();

        // Shuffle pieces
        let mut pieces: Vec<u8> = (0 .. N_PIECES as u8).collect();
        pieces.shuffle(&mut rng);

        Board { board: [INVALID; N_TILES], pieces, empty_tiles: (0 .. N_TILES as u8).collect(), piece: INVALID }
    }

    #[getter]
    fn empty_tiles(&self, py: Python) -> Py<PyArray1<u8>> {
        self.empty_tiles.clone().into_pyarray(py).to_owned().into()
    }

    #[getter]
    fn board(&self, py: Python) -> Py<PyArray1<u8>> {
        self.board.clone().to_vec().into_pyarray(py).to_owned().into()
    }

    fn one_hot(&self, py: Python) -> Py<PyArray1<u8>> {
        let mut arr: Vec<u8> = vec![0; N_TILES * 3 * 3];
        
        for (t, piece_idx) in self.board.iter().enumerate() {
            if *piece_idx == INVALID { continue; };
            let piece = PIECES[*piece_idx as usize];

            for orientation in 0 .. 3 {
                let line_idx = match orientation {
                    0 => (piece[0] - 1) / 4,
                    1 => if piece[1] == 2 { 0 } else { piece[1] - 5 }
                    2 => if piece[2] == 8 { 2 } else { piece[2] - 3 }
                    _ => panic!("error")
                };
                arr[t * 9 + orientation * 3 + line_idx as usize] = 1;
            }
        }
        
        arr.into_pyarray(py).to_owned().into()
    }

    pub fn one_hot_rust(&self) -> Vec<u8> {
        let mut arr: Vec<u8> = vec![0; N_TILES * 3 * 3];
        
        for (t, piece_idx) in self.board.iter().enumerate() {
            if *piece_idx == INVALID { continue; };
            let piece = PIECES[*piece_idx as usize];

            for orientation in 0 .. 3 {
                let line_idx = match orientation {
                    0 => (piece[0] - 1) / 4,
                    1 => if piece[1] == 2 { 0 } else { piece[1] - 5 }
                    2 => if piece[2] == 8 { 2 } else { piece[2] - 3 }
                    _ => panic!("error")
                };
                arr[t * 9 + orientation * 3 + line_idx as usize] = 1;
            }
        }
        
        arr
    }

    pub fn one_hot_delta(&self, one_hot: Vec<u8>, piece_idx: u8, idx: u8) -> Vec<u8> {
        let mut arr = one_hot.clone();
        let tile = &mut arr[idx as usize * 9 .. idx as usize * 9 + 9];

        let piece = PIECES[piece_idx as usize];

        for orientation in 0 .. 3 {
            let line_idx = match orientation {
                0 => (piece[0] - 1) / 4,
                1 => if piece[1] == 2 { 0 } else { piece[1] - 5 }
                2 => if piece[2] == 8 { 2 } else { piece[2] - 3 }
                _ => panic!("error")
            };
            tile[orientation as usize * 3 + line_idx as usize] = 1;
        }

        arr
    }

    pub fn score_change(&self, tile_idx: u8) -> u32 {
        let mut score: u32 = 0;

        for (orientation, rule_idx) in Self::TILE_LINES[tile_idx as usize].iter().enumerate() {
            let indeces: Vec<u8> = Self::LINES[orientation][*rule_idx as usize].iter()
                .filter(|&&i| i != INVALID)
                .map(|&i| self.board[i as usize])
                .collect();
            
            let pieces: Vec<u8> = indeces.iter()
                .map(|&p| if p != INVALID { PIECES[p as usize][orientation] } else { INVALID })
                .collect();

            if pieces.iter().all(|&p| p == pieces[0]) && pieces[0] != INVALID {
                score += pieces[0] as u32 * indeces.len() as u32;
            }
        }

        return score
    }

    pub fn score(&self) -> u32 {
        let mut score: u32 = 0;

        for (orientation, rules) in Self::LINES.iter().enumerate() {
            for line in rules.iter() {
                let indeces: Vec<u8> = line.iter()
                    .filter(|&&i| i != INVALID)
                    .map(|&i| self.board[i as usize])
                    .collect();
                
                let pieces: Vec<u8> = indeces.iter()
                    .map(|&p| if p != INVALID { PIECES[p as usize][orientation] } else { INVALID })
                    .collect();
                
                if pieces.iter().all(|&p| p == pieces[0]) && pieces[0] != INVALID {
                    score += pieces[0] as u32 * indeces.len() as u32;
                }
            }
        }

        return score
    }

    pub fn draw(&mut self) -> u8 {
        match self.pieces.pop() {
            Some(piece) => { self.piece = piece; piece }
            _ => panic!("Cannot draw from empty stack.")
        }
    }

    pub fn play_(&mut self, tile_idx: u8) {
        if tile_idx as usize >= N_TILES { panic!("Cannot place piece at tile_idx={}.", tile_idx) }
        if self.piece == INVALID { panic!("Cannot place invalid piece") }

        // Place piece on board
        self.board[tile_idx as usize] = self.piece;

        // Remove tile index from empty tiles list
        if let Some(pos) = self.empty_tiles.iter().position(|&t| t == tile_idx) {
            self.empty_tiles.remove(pos);
        }
    }

    pub fn play(&mut self, piece: u8, tile_idx: u8) {
        if tile_idx as usize >= N_TILES { panic!("Cannot place piece at tile_idx={}.", tile_idx) }

        // Place piece on board
        self.board[tile_idx as usize] = piece;

        // Remove tile index from empty tiles list
        if let Some(pos) = self.empty_tiles.iter().position(|&t| t == tile_idx) {
            self.empty_tiles.remove(pos);
        }
    }
}

#[pyfunction]
pub fn create_board() -> Board {
    Board::new()
}