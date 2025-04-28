cd bindings
maturin build -r

pip install target/wheels/rust_takeiteasy-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl --force