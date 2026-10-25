$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($file in @('build/bin/matrix_tests.exe','build/bin/matrix_benchmark.exe')) {
    if (!(Test-Path $file)) { throw "Missing $file. Run build.cmd first." }
}
$dest = Join-Path 'results' (Get-Date -Format 'yyyyMMdd-HHmmss')
if (Test-Path $dest) { throw 'Result folder already exists; retry in a second.' }
New-Item -ItemType Directory $dest | Out-Null
try {
    $commit = & git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'git failed' }
    $commit | Set-Content -Encoding utf8 "$dest/commit.txt"
    & git status --short -- . | Set-Content -Encoding utf8 "$dest/source-status.txt"
    if ($LASTEXITCODE -ne 0) { throw 'git status failed' }
    & nvcc --version 2>&1 | Out-File -Encoding utf8 "$dest/nvcc.txt"
    if ($LASTEXITCODE -ne 0) { throw 'nvcc not found' }
    & nvidia-smi 2>&1 | Out-File -Encoding utf8 "$dest/nvidia-smi.txt"
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        & cmake --version | Out-File -Encoding utf8 "$dest/cmake.txt"
    }
    Get-ChildItem build/CMakeFiles -Recurse -Filter '*Compiler.cmake' |
        Select-String -Pattern 'set\(CMAKE_(CXX|CUDA)_COMPILER(_VERSION|_ID)? ' |
        Out-File -Encoding utf8 "$dest/compilers.txt"
    & python --version 2>&1 | Out-File -Encoding utf8 "$dest/python.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Python not found' }
    Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors |
        Format-List | Out-File -Encoding utf8 "$dest/cpu.txt"
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version |
        Format-List | Out-File -Encoding utf8 "$dest/os.txt"
    & cmd /d /c ".\build\bin\matrix_tests.exe --gtest_output=xml:$dest/tests.xml 2>&1" |
        Tee-Object -FilePath "$dest/tests-console.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Benchmark was not run.' }
    & cmd /d /c ".\build\bin\matrix_benchmark.exe --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_out=$dest/benchmarks.json --benchmark_out_format=json 2>&1" |
        Tee-Object -FilePath "$dest/benchmark-console.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
    & python scripts/validate_results.py $dest
    if ($LASTEXITCODE -ne 0) { throw 'Results validation failed' }
    $dest | Set-Content -Encoding ascii results/latest.txt
    Write-Host "SUCCESS. Results: $dest"
} finally {
    Compress-Archive -Path "$dest/*" -DestinationPath "$dest.zip"
    Write-Host "Archive: $dest.zip"
}
