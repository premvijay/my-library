@echo off
REM Check if we're in the correct environment
if not defined VSCMD_ARG_TGT_ARCH (
    echo This script must be run from x64 Native Tools Command Prompt for VS 2022
    exit /b 1
)

REM Ensure we're using x64 tools
if not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    echo This script must be run from x64 Native Tools Command Prompt for VS 2022
    exit /b 1
)

REM Get Python paths and configuration using Python itself
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_paths()['include'])"') do set PYTHON_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"') do set PYTHON_LIBDIR=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set PY_SUFFIX=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))"') do set PY_VERSION=%%i

REM Remove dots from Python version for lib name
set PY_LIB_VERSION=%PY_VERSION:.=%

echo Python version: %PY_VERSION%
echo Python lib version: %PY_LIB_VERSION%
echo Python libdir: %PYTHON_LIBDIR%

REM Set pybind11 include path - assuming it's installed in site-packages
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND_INCLUDE=%%i

echo Python include path: %PYTHON_INCLUDE%
echo Python libs path: %PYTHON_LIBS%
echo Pybind11 include path: %PYBIND_INCLUDE%

REM Get Python executable directory for DLL path
for /f "tokens=*" %%i in ('python -c "import os, sys; print(os.path.dirname(sys.executable))"') do set PYTHON_DLL_DIR=%%i

echo Python DLL directory: %PYTHON_DLL_DIR%

REM Compile the extension
cl.exe /O2 /MD /std:c++17 /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS ^
    /I"%PYTHON_INCLUDE%" /I"%PYBIND_INCLUDE%" ^
    /LD src\grid_assign_pybind11.cpp ^
    /link /LIBPATH:"%PYTHON_LIBDIR%" ^
    /LIBPATH:"%PYTHON_LIBDIR%\libs" ^
    /LIBPATH:"%PYTHON_DLL_DIR%" ^
    /LIBPATH:"%PYTHON_DLL_DIR%\libs" ^
    python%PY_LIB_VERSION%.lib ^
    /OUT:"particles_to_grid%PY_SUFFIX%"

if errorlevel 1 (
    echo Compilation failed
    exit /b 1
)

echo Compilation successful

REM Compile the extension
cl.exe /O2 /MD /std:c++17 /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS ^
    /I"%PYTHON_INCLUDE%" /I"%PYBIND_INCLUDE%" ^
    /LD src\particle_selection_pybind11.cpp ^
    /link /LIBPATH:"%PYTHON_LIBDIR%" ^
    /LIBPATH:"%PYTHON_LIBDIR%\libs" ^
    /LIBPATH:"%PYTHON_DLL_DIR%" ^
    /LIBPATH:"%PYTHON_DLL_DIR%\libs" ^
    python%PY_LIB_VERSION%.lib ^
    /OUT:"select_particles%PY_SUFFIX%"

if errorlevel 1 (
    echo Compilation failed
    exit /b 1
)

echo Compilation successful
