#!/usr/bin/env bash
# Front door: opens the start panel (tools/start_panel.py), which
# launches BeamNG + the control panel. Run from anywhere.
cd "$(dirname "$(readlink -f "$0")")"
exec .venv/bin/python3 tools/start_panel.py "$@"
