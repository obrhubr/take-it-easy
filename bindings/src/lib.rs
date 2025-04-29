use pyo3::prelude::*;

mod board;
mod batched;

#[pymodule]
fn rust_takeiteasy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Simple board implementation
    m.add("N_TILES", board::N_TILES)?;
    m.add("N_PIECES", board::N_PIECES)?;
    m.add_class::<board::Board>()?;
    m.add_function(wrap_pyfunction!(board::create_board, m)?)?;

    // Batched version
    m.add_class::<batched::BatchedBoard>()?;
    m.add_function(wrap_pyfunction!(batched::create_batched, m)?)?;
    Ok(())
}