@echo off
rem Install the in-game control panel mod into BeamNG's userfolder.
rem Usage: beamng_mod\install.bat [userfolder]
rem Default: the newest version folder under %LOCALAPPDATA%\BeamNG.drive
rem (userfolder location since 0.32), falling back to the old
rem Documents\BeamNG.drive location.
setlocal
set "HERE=%~dp0"
set "USERFOLDER=%~1"
if not "%USERFOLDER%"=="" goto have

for /f "delims=" %%D in ('dir /b /ad /on "%LOCALAPPDATA%\BeamNG.drive" 2^>nul') do (
    if exist "%LOCALAPPDATA%\BeamNG.drive\%%D\mods" set "USERFOLDER=%LOCALAPPDATA%\BeamNG.drive\%%D"
)
if not "%USERFOLDER%"=="" goto have
if exist "%USERPROFILE%\Documents\BeamNG.drive\mods" set "USERFOLDER=%USERPROFILE%\Documents\BeamNG.drive"

:have
if "%USERFOLDER%"=="" (
    echo userfolder not found under %%LOCALAPPDATA%%\BeamNG.drive or
    echo Documents\BeamNG.drive — pass it explicitly:
    echo   beamng_mod\install.bat ^<userfolder^>
    exit /b 1
)
if not exist "%USERFOLDER%" (
    echo userfolder not found: "%USERFOLDER%"
    exit /b 1
)

set "DEST=%USERFOLDER%\mods\unpacked\onnx-panel"
xcopy /e /i /y /q "%HERE%onnx-panel" "%DEST%" >nul
if errorlevel 1 (
    echo copy failed
    exit /b 1
)
echo installed -^> %DEST%
echo restart BeamNG so the mod mounts; the control panel loads it
echo automatically (or run: extensions.load('onnxPanel') in the console)
