### MLIRSat: Equality Saturation for MLIR

Use `python3 -m bench.bench` to run the benchmarks or `python3 main.py` to run the program. 


The MLIR files used for evaluation are in the [data/mlir](data/mlir/), and their egg equivalents are in [data/eggs](data/eggs/). The optimized MLIR and egg equivalents are in the [data/converted](data/converted/) directory.

The [mlir/parser.py](mlir/parser.py) and [eggie/parser.py](eggie/parser.py) files contain the logic to convert from MLIR to Egglog and Egglog to MLIR respectively.

