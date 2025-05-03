# Optimised Take It Easy

This branch implements an optimised approach to training, including `rust` bindings for Take It Easy.

## Requirements

Install `cargo` and `rust` then run `pip3 install -r requirements.txt`. The package `maturin` is required for the bindings.

## Usage

Compile the `rust` bindings by running `make.sh` (make it executable using `chmod +x make.sh` before). This should automatically install the wheel into your current environment.

You can then start training with `python3 nn.py` or configure the hyperparameters first.

## Why?

The `rust` version implements a `BatchedBoard` class, which allows playing `n` games at the same time. This speeds up training massively, as the bottle-neck (inference on the model) is eliminated through batch inference.

The other speed-up is implementing *incremental one-hot encoding*. By only modifying the one-hot encoding for the tile the piece was placed on, is faster than always starting from 0.