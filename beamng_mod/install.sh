#!/usr/bin/env bash
# Install the in-game control panel mod into BeamNG's userfolder.
# Usage: bash beamng_mod/install.sh [userfolder]
# Default userfolder matches launch_beamng.sh (tech mode on Linux).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
USERFOLDER="${1:-$HOME/.local/share/BeamNG/BeamNG.tech/current}"
DEST="$USERFOLDER/mods/unpacked/onnx-panel"

if [ ! -d "$USERFOLDER" ]; then
  echo "userfolder not found: $USERFOLDER" >&2
  echo "pass it explicitly: bash beamng_mod/install.sh <userfolder>" >&2
  exit 1
fi

mkdir -p "$DEST"
cp -r "$HERE/onnx-panel/." "$DEST/"
echo "installed -> $DEST"
echo "restart BeamNG so the mod mounts; the control panel loads it"
echo "automatically (or run: extensions.load('onnxPanel') in the console)"
