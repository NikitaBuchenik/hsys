# Практическая работа №2: умножение матриц CUDA

**Статус:** 10 октября 2026 выполнены сборка и запуск на RTX 2060.
Все 351 тест прошли (в том числе 343 сочетания размеров). Сохранены семь повторов
для каждого из семи размеров и обеих реализаций. Итоговый отчет: [report.pdf](report.pdf).
Исходные измерения, XML тестов, таблица и графики: [experiment/](experiment/).
`report-draft.pdf` оставлен как прежний проект; для сдачи используйте `report.pdf`.

При n=1024 медианы: CUDA 3.851325 мс, Eigen 72.532675 мс; отношение 18.833.
CUDA быстрее начиная с n=64 среди проверенных размеров. Это время вычислений
относительно однопоточного Eigen, без передачи матриц CPU/GPU.

Измеренный коммит: `34211d8ec2e1150803b7db2b7a6d282e0554d795`.
Перед замерами применен `editbin /STACK:67108864 build/bin/matrix_benchmark.exe`:
исходная настройка CMake была проигнорирована компилятором. Теперь параметр
передается через `HOST_LINK:LINKER`; исправленную сборку отдельно еще не запускали.
Compiler в nvcc.txt: 13.0.88, runtime в JSON: 13020. Различие версий сохранено,
причина не установлена. Модель CPU не сообщалась; не выводилась из кешей/частоты.

Авторы: А.А. Платонов, Н.С. Бученик, КИ23-06Б. Преподаватель: С.А. Тарасов.

## Что реализовано

- `core/`: RAII `Data`, тривиально-копируемый `MatrixView`, фасад `Matrix` со
  `shared_ptr`, row-major, наивный 2D-кернел 16×16 без shared memory, `operator*`.
- `tests/`: Google Test + Eigen::MatrixXf; 343 сочетания m,n,k из
  `{1,2,3,127,128,129,512}` и дополнительные граничные/архитектурные проверки.
- `benchmarks/`: Google Benchmark для `{16,32,64,128,256,512,1024}`; реальный
  `operator*`, CUDA Events охватывают только кернел, Eigen - один CPU-поток.
- `scripts/`: сборка, запуск, сохранение сырых результатов, графики Plotly,
  создание PDF по оформлению первой работы.

## Подготовка компьютера с RTX 2060 (Windows 10/11, x64)

RTX 2060 имеет compute capability 7.5, поэтому скрипт использует архитектуру `75`.

1. Установить актуальный [драйвер NVIDIA](https://www.nvidia.com/Download/index.aspx).
2. Установить [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
   или Visual Studio 2022, компонент **«Разработка классических приложений на C++»**:
   MSVC v143, Windows SDK, инструменты CMake для Windows. Выбрать совместимую с
   CUDA версию MSVC; в документации CUDA 13.0.1 указана серия MSVC 193x.
   При необходимости установить дополнительный toolset v14.39 через Individual components.
3. Установить [CUDA Toolkit 13.0 Update 1](https://developer.nvidia.com/cuda-13-0-1-download-archive).
   Драйвер должен удовлетворять требованиям выбранной версии Toolkit.
   Перезапустить терминал после установки.
4. Установить [Git](https://git-scm.com/downloads/win), [Python 3.12](https://www.python.org/downloads/)
   с добавлением в PATH и Google Chrome (для экспорта Plotly в PNG).
5. Открыть **Developer PowerShell for VS 2022 (x64)**. Если CUDA сообщает о
   неподдерживаемом компиляторе, выбрать установленный toolset 14.39 через
   `vcvarsall.bat x64 -vcvars_ver=14.39` в Developer Command Prompt,
   затем запустить `powershell` из этого окна. Не применять `--allow-unsupported-compiler`.

В терминале:

```powershell
git clone https://github.com/NikitaBuchenik/hsys.git
cd hsys/second
python -m pip install "cmake>=3.24,<4" ninja -r scripts/requirements.txt
nvidia-smi
nvcc --version
cl
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

`ExecutionPolicy Bypass` относится только к этому процессу запуска локального
скрипта; системная политика не меняется. При установленном PowerShell с разрешенным
запуском локальных скриптов достаточно `./scripts/run.ps1`.

Первый запуск скачивает зафиксированные версии Eigen 3.4.0, Google Test 1.15.2
и Google Benchmark 1.9.1. Нужен интернет. Сборка использует C++20 + CMake + Ninja.
При ошибке скрипт останавливается. Пришлите текст ошибки, а не меняйте точность тестов.

## Результат запуска

В новой папке `results/ГГГГММДД-ЧЧММСС/` сохраняются:

- `tests.xml` - настоящий протокол тестов;
- `benchmarks.json` - все повторы Google Benchmark и окружение;
- `nvidia-smi.txt`, `nvcc.txt`, `commit.txt` - оборудование и версии;
- `measurements.csv`, `summary.json` - обработанные измерения;
- `complexity.html/png`, `speedup.html/png` - графики только по этим измерениям;
- `report.pdf` - отчет с таблицей, графиками и анализом.

Скрипт упаковывает файлы в `results/work2-<дата>.zip`. Пришлите архив для проверки
и добавления окончательного отчета в репозиторий. `results/` исключена из Git,
чтобы случайно не смешивать измерения разных запусков. Перед публикацией
отчета проверить графики, аппаратные сведения и выводы.

Если экспорт PNG не удался, сырые результаты не теряются. Установите Chrome и повторите:

```powershell
python scripts/plot.py results/<папка>/benchmarks.json
python scripts/make_report.py --results results/<папка> --output results/<папка>/report.pdf
```

## Методика и ограничения

Обычный `a*b` выделяет результат и синхронно возвращает готовую матрицу. Во время
CUDA-бенчмарка `ScopedKernelTimer` включает два заранее созданных события внутри
пути оператора. Выделение/копирование/освобождение памяти не попадает в интервал.
События и кернел используют default stream. Это compute-only время, не end-to-end.

Eigen вычисляет `c.noalias() = a*b` в заранее выделенный результат. Установлен
`EIGEN_RUNTIME_NO_MALLOC`; проверка активна даже в Release. Внутренние буферы
упаковки используют стек (лимит Eigen 8 МиБ, стек процесса 64 МиБ). Если Eigen
попытается выделить heap-память, измерение завершается ошибкой вместо публикации
некорректного времени. Стековые служебные действия и упаковка входят во время CPU.

Прогрев: 3 умножения. Повторы: 7, минимум 0.2 секунды на серию. Медиана по
повторам, для времени показаны min/max. Ускорение = median(Eigen)/median(CUDA).
Eigen намеренно однопоточный. Не запускать игры и другие GPU-задачи параллельно.

`isApprox` - относительное сравнение норм, хотя задание называет точность
абсолютной. Тесты проверяют и `isApprox(..., 1e-5f)`, и максимальную абсолютную
ошибку `<=1e-5`. Основной набор: воспроизводимые значения q/16, q∈[-8,8], точно
представимые суммы для всех заданных k. Дополнительный тест использует обычные
случайные float малой амплитуды. Это не обещание абсолютной ошибки 1e-5 для
произвольных по величине float.

`Matrix` поддерживает float/double для умножения; классы Data/View шаблонные.
Копирование Matrix разделяет память, копирование Data - глубокое. Методы `data()`
и `view()` предоставлены согласно UML; не заменять вручную владельца или
представление на несогласованные данные. После move размеры нулевые, обращение
к `data()` бросает исключение. Индексация View допустима только для валидных индексов.

## Linux

Установить совместимые драйвер, CUDA Toolkit, C++-компилятор, CMake ≥3.24,
Ninja, Git, Python и DejaVu Fonts. Установить Python-зависимости, затем:

```bash
bash scripts/run.sh
```

Проверка памяти (дополнительно, при наличии Compute Sanitizer):

```text
compute-sanitizer --tool memcheck --leak-check full build/bin/matrix_tests.exe
```

## Оформление отчета

Титульные данные сохранены по образцу первой работы; номер работы изменен на 2.
Проект PDF можно пересобрать без GPU:

```text
python scripts/make_report.py --draft --output report-draft.pdf
```

Генерация окончательного отчета требует успешного XML тестов (включая все 343
случая), полного JSON замеров, соответствующего summary и двух PNG графиков.
Исходное задание требует СТУ 7.5-07-2021; оформление ориентировано на переданный
образец, полная формальная проверка стандарта без его текста не заявляется.

Справка: [CUDA Windows](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-installation-guide-microsoft-windows/index.html),
[видеокарты NVIDIA](https://developer.nvidia.com/cuda/gpus),
[Eigen isApprox](https://libeigen.gitlab.io/eigen/docs-3.4/classEigen_1_1DenseBase.html),
[Google Benchmark](https://google.github.io/benchmark/user_guide.html).
