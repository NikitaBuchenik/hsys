param([string]$Architecture = 'native')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vcvars = 'H:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat'
    if (!(Test-Path -LiteralPath $vcvars)) {
        $finder = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (!(Test-Path -LiteralPath $finder)) { throw 'Install VS 2022 C++ Build Tools with MSVC 14.39, Windows SDK and CMake tools.' }
        $installation = & $finder -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (!$installation) { throw 'VS 2022 C++ tools not found.' }
        $vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvarsall.bat'
    }
    # Import the environment into this process; quoted paths may contain spaces.
    $environment = & cmd.exe /d /c "call `"$vcvars`" x64 -vcvars_ver=14.39 >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'MSVC 14.39 initialization failed. Install that individual component in VS Installer.' }
    foreach ($line in $environment) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
}
foreach ($tool in @('cl.exe','nvcc.exe','cmake.exe','ninja.exe','git.exe')) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing $tool. Check CUDA Toolkit / VS CMake components / Git." }
}
& cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build build --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Host 'BUILD SUCCESS. Next: run.cmd (PowerShell: .\run.cmd)'
