#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Eigen scratch buffers are on the stack, not the heap.
ulimit -s 65536
result="results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$result"
nvidia-smi | tee "$result/nvidia-smi.txt"
nvcc --version > "$result/nvcc.txt"
git rev-parse HEAD > "$result/commit.txt"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel 2
build/bin/matrix_tests --gtest_output="xml:$result/tests.xml"
build/bin/matrix_benchmark --benchmark_min_time=0.2s --benchmark_repetitions=7 --benchmark_out="$result/benchmarks.json" --benchmark_out_format=json
python3 scripts/plot.py "$result/benchmarks.json"
python3 scripts/make_report.py --results "$result" --output "$result/report.pdf"
echo "Send the entire $result directory back for review."
