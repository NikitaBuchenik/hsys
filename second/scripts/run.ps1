param([int]$Jobs=2)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
foreach ($tool in @('git','cmake','ninja','nvcc','cl','python','nvidia-smi')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing $tool. Follow README.md; run from Developer PowerShell for VS 2022."
    }
}
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$result=Join-Path 'results' $stamp
New-Item -ItemType Directory -Force $result | Out-Null
Run-Checked 'nvidia-smi' @()
& nvidia-smi | Out-File -Encoding utf8 (Join-Path $result 'nvidia-smi.txt')
& nvcc --version | Out-File -Encoding utf8 (Join-Path $result 'nvcc.txt')
& git rev-parse HEAD | Out-File -Encoding utf8 (Join-Path $result 'commit.txt')
Run-Checked 'cmake' @('-S','.','-B','build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75')
Run-Checked 'cmake' @('--build','build','--parallel',"$Jobs")
Run-Checked '.\build\bin\matrix_tests.exe' @("--gtest_output=xml:$result/tests.xml")
Run-Checked '.\build\bin\matrix_benchmark.exe' @('--benchmark_min_time=0.2s','--benchmark_repetitions=7',"--benchmark_out=$result/benchmarks.json",'--benchmark_out_format=json')
Run-Checked 'python' @('scripts/plot.py',"$result/benchmarks.json")
Run-Checked 'python' @('scripts/make_report.py','--results',$result,'--output',"$result/report.pdf")
$archive="results/work2-$stamp.zip"
Compress-Archive -Path "$result/*" -DestinationPath $archive
Write-Host "Done. Send $archive back for review."
