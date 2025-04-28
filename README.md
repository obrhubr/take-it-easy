# Optimised Take It Easy

This branch implements an optimised approach to training, including `rust` bindings for Take It Easy.

## Why?

The `rust` version implements a `BatchedBoard` class, which allows playing `n` games at the same time. This speeds up training massively, as the bottle-neck (inference on the model) is eliminated through batch inference.

The other speed-up is implementing *incremental one-hot encoding*. By only modifying the one-hot encoding for the tile the piece was placed on, is faster than always starting from 0.

## How much faster is it?

A single iteration takes about `20s` to generate the training date (16384 games) on a 16 core VM and about `1min30s` to train the model on the CPU.

The unoptimised, non-batched python version took more then `10min` to generate the training data and the same time to train on the CPU.