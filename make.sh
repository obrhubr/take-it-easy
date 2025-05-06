cd bindings
maturin build -r

pip install target/wheels/*.whl --force