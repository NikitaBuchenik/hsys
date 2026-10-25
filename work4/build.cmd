@echo off
setlocal
cd /d "%~dp0"
where cl >nul 2>nul
if not errorlevel 1 goto configure
if exist "H:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat" (
    call "H:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
    if errorlevel 1 exit /b 1
) else (
    echo Open an x64 VS 2022 Developer Command Prompt with MSVC 14.39 first.
    exit /b 1
)
:configure
where nvcc >nul 2>nul
if errorlevel 1 exit /b 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
if errorlevel 1 exit /b 1
cmake --build build --parallel 2
exit /b %errorlevel%
