# Beamng-onnx

Drive **BeamNG.tech** with comma.ai's openpilot driving models — no
openpilot install, just the ONNX files and onnxruntime.

This is a Linux-only fork of [140er/simsteer](https://github.com/140er/simsteer)
rebuilt around BeamNG. Upstream drives ETS2 / Forza / Assetto Corsa on
Windows by capturing the game window and steering through a virtual
gamepad; this fork keeps simsteer's model core (warp, preprocessing,
decode, learners) and replaces all of the I/O:

| | upstream simsteer (Windows) | this fork (Linux + BeamNG) |
|---|---|---|
| Frames | dxcam/mss screen capture | beamngpy `Camera` sensor, shared-memory streaming |
| Telemetry | per-game plugins | beamngpy `Electrics` |
| Control | ViGEm / vJoy virtual devices | `vehicle.control()` |
| Inference | onnxruntime DirectML | onnxruntime **ROCm** (AMD GPU), CPU fallback |

The camera sensor renders the model's windshield view independently of
what's on screen — you can film the AI car from any angle (free cam,
chase cam) while it drives, or drive a second vehicle around it.

## Status

- **M1 — model sees BeamNG** ✅ lane probs peak ~1.0 while driving,
  plan reach 100–240 m, verified warp geometry (comma-style windshield
  mount, FOV checked against horizon + model lane widths)
- **Live viewer** ✅ `tools/live_view.py` — simsteer's overlay (plan,
  lane lines, road edges) on the live camera at 20 Hz
- **GPU inference** ✅ ROCm on RDNA4 (gfx1201): full pipeline
  7.5 ms/step (133 Hz capable)
- **Supercombo** ✅ runs openpilot master's re-unified
  `driving_supercombo.onnx` (2026) with `--supercombo`
- **M2 — solid 20 Hz loop** ✅ shared-memory frame streaming
  (0.13 ms/frame vs 12 ms polled)
- **M3 — closed loop** ⏳ next: model curvature + plan accel →
  `vehicle.control()` (lateral + end-to-end longitudinal)
- **M4 — scenarios** ⏳ lead-car cut-in tests with a player-driven
  second vehicle, construction-site props, filming runs

## Requirements

- **BeamNG.tech** (research license — the `Camera` sensor needs it),
  launched with the tech server: `bash launch_beamng.sh`
- Any Linux distro — **no container required**. Python 3.12 venv:

  ```bash
  uv venv .venv --python 3.12        # or python3 -m venv .venv
  uv pip install -r requirements.txt
  ```

  CPU inference runs the full pipeline at ~60 Hz, 3× the 20 Hz it
  needs — the GPU section below is optional.

  (This repo is developed on Bazzite with the Python env in a
  distrobox, purely because that's an easy place to put the ROCm
  libraries on an immutable OS. Dev-machine detail, not a dependency.)

### Models (not in the repo)

Put the ONNX files in `models/`:

- **Split pair** (openpilot 0.11.1): `driving_vision.onnx` +
  `driving_policy.onnx` — copy from an openpilot checkout at a 0.11.x
  tag (`selfdrive/modeld/models/`, git-lfs).
- **Supercombo** (openpilot master, 2026): `driving_supercombo.onnx` —
  same path on current master. Output layout is read from the file's
  own metadata, so checkpoint swaps don't need code changes.

### ROCm (optional, AMD GPUs)

`onnxruntime-rocm` from PyPI plus the ROCm 6.4 runtime libs
(`hipblas rocblas miopen-hip hip-runtime-amd hipfft hipsparse hiprand
rocrand rccl roctracer hipsolver rocsolver rocfft` from
repo.radeon.com), then `echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf
&& ldconfig`. Falls back to CPU automatically (~60 Hz there, still fine).

## Running

**Easiest path — the start panel:**

```bash
python tools/start_panel.py
```

It auto-detects your BeamNG install and tech.key, lets you pick a map
and vehicle, launches the game with the tech server if it isn't up,
and starts the control panel. `west_coast_usa` uses our scripted
scenario with the calibrated spawn; any other map runs through
*freeroam interception* — load the map in-game, enter a car, press
START and the model hooks that car. Without a BeamNG.tech `tech.key`
the planned player-view + virtual-gamepad mode isn't built yet, so a
tech license (free for personal use from BeamNG) is currently
required.

**Manual pieces:**

```bash
bash launch_beamng.sh                     # host: BeamNG.tech + tech server

# in the Python env:
python tools/control_panel.py             # the main app (viewer+telemetry)
python tools/control_panel.py --attach    # hook the car already in-game
python tools/live_view.py --supercombo    # live overlay viewer; drive manually
python tools/live_view.py --ai            # let BeamNG's AI drive (split model)
python tools/m1_beamng_frame.py --supercombo --ai --seconds 20   # scored probe
python tools/camera_probe.py              # camera-mount/FOV verification shots
```

`beamng/world.py` holds the scenario/vehicle/camera constants (mount
position, FOV, `lateral_sign=+1` — BeamNG renders unmirrored, unlike
upstream's screen-capture games).

### In-game control panel (mod)

`beamng_mod/` ships a tiny GE-Lua mod that draws an imgui window inside
BeamNG with ENGAGE / lane-change / turn / LONG / CAL buttons, relayed
over localhost UDP to `tools/control_panel.py` (which sends live status
back). Install once, then restart the game:

```bash
bash beamng_mod/install.sh    # copies into the userfolder's mods/unpacked/
```

The control panel auto-loads the extension on startup; clicks work from
either the in-game window or the panel window.

## Upstream docs

simsteer's [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) still describes
the model core accurately (what's a comma port vs simsteer's own).
SETUP/TUNING are Windows-specific and don't apply here.

MIT, same as upstream. Models are comma.ai's.
