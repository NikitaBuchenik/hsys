# Work 3 — CUDA matrix multiplication with shared memory

Практическая работа 3 для репозитория hsys. Каталог автономный: предыдущие работы не меняются.

## Реализация
- `kernel_matmul_shmem` использует плитки 16×16 в shared memory, загрузку с обработкой краёв и `__syncthreads()`.
- `MatrixView` передаётся в kernel по значению.
- `operator*` выбирает стратегию SharedMemory. Старая наивная реализация оставлена как `multiply_naive` для сравнения.
- Google Test создаёт все 7³ = 343 сочетания размеров и сравнивает результат с Eigen через `isApprox(..., 1e-5f)`; дополнительно проверяются некратные плитке размеры, нулевые/несовместимые размеры.
- Google Benchmark запускает обе реализации. CUDA Events измеряют только kernel, выделение памяти и копирование host/device не попадают в измеряемый интервал.
- `scripts/plot.py` строит графики только по реальным измерениям JSON; подставных результатов нет.

## Требования
CUDA Toolkit (nvcc), совместимый NVIDIA GPU и драйвер, CMake ≥ 3.24, Ninja, C++20 toolchain, Python 3.10+.

## Windows (Developer PowerShell for VS 2022)
```powershell
cd work3
python -m pip install -r scripts/requirements.txt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
.\build\bin\matrix_tests.exe
.\build\bin\matrix_benchmark.exe --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_out=results/benchmarks.json --benchmark_out_format=json
python scripts/plot.py results/benchmarks.json
```
Перед benchmark создайте каталог `results`. Архитектуру GPU при необходимости задайте явно, например `-DCMAKE_CUDA_ARCHITECTURES=75`.

## Linux
```bash
cd work3
python3 -m pip install -r scripts/requirements.txt
mkdir -p results
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
build/bin/matrix_benchmark --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_out=results/benchmarks.json --benchmark_out_format=json
python3 scripts/plot.py results/benchmarks.json
```

## Методика и анализ
Умножение квадратных матриц имеет теоретическую сложность Θ(n³). Тайлинг сокращает повторные обращения к глобальной памяти: каждая плитка A/B загружается в shared memory и используется повторно несколькими FMA-операциями. Для малых n запуск kernel может доминировать над вычислениями, поэтому ускорение не обязано быть положительным на всех размерах. Для отчёта используйте реальные CSV/PNG/HTML из `results`, укажите модель GPU, версию CUDA, компилятор и результаты тестов. Не трактуйте теоретическую Θ(n³) как гарантированную форму времени на конечном диапазоне размеров.
