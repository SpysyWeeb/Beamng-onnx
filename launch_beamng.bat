@echo off
rem Launches BeamNG directly with the tech server flags.
rem -nosteam bypasses Steam auth so no Steam session is needed.
rem -tcom -tport 64256 start the tech server that beamngpy connects to.
rem
rem /belownormal (the nice +10 of Windows): modeld runs a CPU-compiled
rem model and must hold 20 Hz; BeamNG saturating all cores starves it,
rem flagging modelV2/cameraOdometry invalid and cascading into commIssue
rem soft-disables. Deprioritizing the game lets modeld win the cycles.
rem
rem The start panel (start.bat) does all of this for you, install-path
rem auto-detection included — this script is the manual fallback.

set "BEAMNG_BIN=C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive\Bin64\BeamNG.drive.x64.exe"

if not exist "%BEAMNG_BIN%" (
    echo BeamNG binary not found: "%BEAMNG_BIN%"
    echo Edit BEAMNG_BIN at the top of this script to point at your
    echo install's Bin64\BeamNG.drive.x64.exe ^(or .tech^).
    pause
    exit /b 1
)

echo [BeamNG] Launching (below-normal priority): "%BEAMNG_BIN%" -nosteam -tcom -tport 64256
start "" /belownormal "%BEAMNG_BIN%" -nosteam -tcom -tport 64256
