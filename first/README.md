# Практическая работа №1 — CUDA Vector

Проект реализует:

- `Data<AtomT>` — RAII-обёртку над глобальной памятью CUDA;
- `VectorView<AtomT>` — лёгкое trivially-copyable представление device-данных;
- `Vector<AtomT>` — фасад на базе `std::shared_ptr<Data<AtomT>>`;
- CUDA kernel `kernel_vecadd`;
- `operator+` для `Vector`;
- Google Test + Eigen;
- Google Benchmark + CUDA Events API;
- Python-скрипт для графиков.

## Важно: локальный ПК без NVIDIA GPU

CUDA-код нельзя нормально выполнить на обычном CPU. На Windows можно писать код и работать с CMake/Git, но сборку и запуск CUDA-части выполняем на машине с NVIDIA GPU.

Для этого проекта удобно использовать Google Colab с GPU.

## Локально на Windows

Проверить инструменты:

```powershell
git --version
cmake --version
ninja --version
```

MSVC проверяется в **Developer PowerShell for VS 2022**:

```powershell
cl
```

Создать Git-репозиторий:

```powershell
git init
git add .
git commit -m "Initial CUDA practical work"
```

## Google Colab

1. Создай новый notebook.
2. Включи GPU:
   `Runtime -> Change runtime type -> T4 GPU` (если доступен).
3. Проверь GPU:

```bash
!nvidia-smi
```

4. Склонируй GitHub-репозиторий:

```bash
!git clone https://github.com/YOUR_LOGIN/YOUR_REPO.git
%cd YOUR_REPO
```

Если репозиторий приватный, удобнее загрузить ZIP через Colab или использовать GitHub authentication.

5. Установи зависимости:

```bash
!apt-get update -qq
!apt-get install -y -qq cmake ninja-build libeigen3-dev
```

6. Проверь CUDA:

```bash
!nvcc --version
```

7. Собери проект:

```bash
!cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j2
```

8. Запусти тесты:

```bash
!ctest --test-dir build --output-on-failure
```

9. Запусти benchmark:

```bash
!./build/benchmarks/vector_benchmark --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

## Benchmark Eigen

Задание требует сравнить CUDA Vector и `Eigen::VectorXf`.

Eigen-часть можно измерять отдельно на той же Colab-машине. Важно не включать в измеряемую область CUDA allocation/copy/free. CUDA-измерение в проекте выполняется через CUDA Events вокруг kernel launch.

## Графики

Собери CSV вида:

```text
n,cuda_seconds,eigen_seconds
8,0.000001,0.000002
64,0.000001,0.000002
512,0.000001,0.000003
```

Установи Python-зависимости:

```bash
pip install -r scripts/requirements.txt
```

Построй графики:

```bash
python scripts/plot.py results.csv
```

Будут созданы:

- `complexity.png`
- `speedup.png`

## Структура

```text
core/
  include/
    cuda_check.cuh
    data.cuh
    vector_view.cuh
    vector.cuh
  src/
    vector.cu

tests/
  src/
    vector_tests.cu

benchmarks/
  src/
    vector_benchmark.cu

scripts/
  plot.py
  requirements.txt
```
