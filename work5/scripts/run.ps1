$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($file in @('build/bin/vector_tests.exe','build/bin/vector_benchmark.exe')) {
    if (!(Test-Path $file)) { throw "Missing $file. Run build.cmd first." }
}
foreach ($tool in @('python','nvcc','nvidia-smi','git')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing $tool. Open a fresh terminal after installation." }
}
$dest = Join-Path 'results' (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory $dest -ErrorAction Stop | Out-Null
try {
    & git rev-parse HEAD | Set-Content -Encoding utf8 "$dest/commit.txt"
    if ($LASTEXITCODE -ne 0) { throw 'git failed' }
    $statusLines = & git status --short -- .
    if ($LASTEXITCODE -ne 0) { throw 'git status failed' }
    [IO.File]::WriteAllText((Join-Path (Get-Location) "$dest/source-status.txt"), ($statusLines -join "`n"))
    & nvcc --version | Out-File -Encoding utf8 "$dest/nvcc.txt"
    if ($LASTEXITCODE -ne 0) { throw 'nvcc failed' }
    & nvidia-smi | Out-File -Encoding utf8 "$dest/nvidia-smi.txt"
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors |
        Format-List | Out-File -Encoding utf8 "$dest/cpu.txt"
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version |
        Format-List | Out-File -Encoding utf8 "$dest/os.txt"
    Get-ChildItem build/CMakeFiles -Recurse -Filter '*Compiler.cmake' |
        Select-String -Pattern 'set\(CMAKE_(CXX|CUDA)_COMPILER(_VERSION|_ID)? ' |
        Out-File -Encoding utf8 "$dest/compilers.txt"
    & python scripts/analyze.py --snapshot $dest
    if ($LASTEXITCODE -ne 0) { throw 'Source snapshot failed' }
    & cmd /d /c ".\build\bin\vector_tests.exe --gtest_output=xml:$dest/tests.xml 2>&1" |
        Tee-Object -FilePath "$dest/tests-console.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; benchmarks were not run.' }
    & cmd /d /c ".\build\bin\vector_benchmark.exe --benchmark_min_time=0.05s --benchmark_repetitions=5 --benchmark_out=$dest/benchmarks.json --benchmark_out_format=json 2>&1" |
        Tee-Object -FilePath "$dest/benchmark-console.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark process failed' }
    & python scripts/analyze.py --validate $dest
    if ($LASTEXITCODE -ne 0) { throw 'Incomplete or invalid experiment' }
    $dest | Set-Content -Encoding ascii results/latest.txt
    Write-Host "SUCCESS. Results: $dest"
    Write-Host 'Next: report.cmd (PowerShell: .\report.cmd)'
} finally {
    Compress-Archive -Path "$dest/*" -DestinationPath "$dest.zip"
    Write-Host "Archive: $dest.zip"
}
