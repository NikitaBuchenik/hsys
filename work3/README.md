# Work 3 — умножение матриц с shared memory

Отчёт, исходный код, тестовый протокол и измерения находятся в этом каталоге.

## Что реализовано
- `kernel_matmul_shmem`: плитки 16×16 в разделяемой памяти и синхронизация `__syncthreads()`.
- Передача `MatrixView` в ядро по значению.
- Стратегии `Naive` и `SharedMemory`; `operator*` выбирает shared-memory реализацию.
- 343 сочетания размеров в Google Test с проверкой по Eigen при допуске (10^{-5}).
- Google Benchmark с CUDA Events; выделение и освобождение памяти не включаются во время ядра.
- Графики сложности и ускорения по сохранённым измерениям.
- Отчёт `results/report.docx`, оформленный по образцу отчёта второй работы.

## Результаты имеющегося эксперимента
Из сохранённого протокола: 347 тестов, 0 ошибок, 0 падений. GPU — NVIDIA GeForce RTX 2060. По медианным измерениям shared-memory реализация достигает ускорения примерно 1,292× при размере 1024×1024. На размерах 16 и 32 наивный вариант быстрее; это согласуется с тем, что накладные расходы тайлинга и синхронизации доминируют при малом объёме вычислений.

## Пересборка и повторение эксперимента
Нужны CUDA Toolkit, совместимый NVIDIA GPU, CMake ≥ 3.24, Ninja, C++20 toolchain и Python 3.10+.

### Windows (Developer PowerShell for VS 2022)
```powershell
cd work3
python -m pip install -r scripts/requirements.txt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
New-Item -ItemType Directory -Force results | Out-Null
.\build\bin\matrix_tests.exe --gtest_output=xml:results/tests.xml
.\build\bin\matrix_benchmark.exe --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_out=results/benchmarks.json --benchmark_out_format=json
python scripts/plot.py results/benchmarks.json
python scripts/make_report.py --results results --output results/report.docx
```

### Linux
```bash
cd work3
python3 -m pip install -r scripts/requirements.txt
mkdir -p results
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
build/bin/matrix_tests --gtest_output=xml:results/tests.xml
build/bin/matrix_benchmark --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_out=results/benchmarks.json --benchmark_out_format=json
python3 scripts/plot.py results/benchmarks.json
python3 scripts/make_report.py --results results --output results/report.docx
```

## Теория
Для матриц (m\times k) и (k\times n) сложность равна (Theta(mkn)); для квадратных матриц — (Theta(n^3)). Shared memory сокращает повторные обращения к глобальной памяти, но добавляет загрузку плиток и барьерную синхронизацию. Поэтому выигрыш зависит от размера матриц и архитектуры GPU.
