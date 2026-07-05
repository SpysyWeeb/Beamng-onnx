@echo off
rem Install the in-game control panel mod into BeamNG's userfolder.
rem Usage: beamng_mod\install.bat [userfolder]
rem Default: every userfolder that exists gets the mod (drive + tech,
rem current and versioned layouts) — it's a tiny UI app, and installing
rem everywhere beats guessing which one the game will mount.
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
set "FOUND="

if not "%~1"=="" (
    call :install "%~1"
    goto done
)

for %%B in ("%LOCALAPPDATA%\BeamNG\BeamNG.drive" "%LOCALAPPDATA%\BeamNG\BeamNG.tech" "%LOCALAPPDATA%\BeamNG.drive" "%LOCALAPPDATA%\BeamNG.tech" "%USERPROFILE%\Documents\BeamNG.drive" "%USERPROFILE%\OneDrive\Documents\BeamNG.drive") do (
    if exist "%%~B\current\mods" call :install "%%~B\current"
    if exist "%%~B\mods" call :install "%%~B"
    for /f "delims=" %%D in ('dir /b /ad "%%~B" 2^>nul') do (
        if not "%%D"=="current" if exist "%%~B\%%D\mods" call :install "%%~B\%%D"
    )
)

:done
if "%FOUND%"=="" (
    echo no userfolder found — pass it explicitly:
    echo   beamng_mod\install.bat ^<userfolder^>
    exit /b 1
)
echo restart BeamNG so the mod mounts; the control panel loads it
echo automatically (or run: extensions.load('onnxPanel') in the console)
exit /b 0

:install
set "DEST=%~1\mods\unpacked\onnx-panel"
xcopy /e /i /y /q "%HERE%onnx-panel" "%DEST%" >nul
if not errorlevel 1 (
    echo installed -^> %DEST%
    set "FOUND=1"
)
exit /b 0
