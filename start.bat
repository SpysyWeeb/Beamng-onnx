@echo off
rem Front door: opens the start panel (tools/start_panel.py), which
rem launches BeamNG + the control panel. Double-clickable, run from anywhere.
pushd "%~dp0"
".venv\Scripts\python.exe" tools\start_panel.py %*
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% neq 0 (
    echo.
    echo Exited with code %EXITCODE%.
    pause
)
