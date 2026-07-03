# Beamng-onnx

Drive **BeamNG** with comma.ai's openpilot driving models — no
openpilot install, just the ONNX files and onnxruntime. The model
steers and works the pedals end-to-end; you watch it drive.

A Linux-only fork of [140er/simsteer](https://github.com/140er/simsteer)
rebuilt around BeamNG. Upstream drives ETS2 / Forza / Assetto Corsa on
Windows; this fork keeps simsteer's model core (warp, preprocessing,
decode, learners) and replaces all of the I/O. There are **two ways to
feed and drive the game:**

| | **tech mode** (BeamNG.tech license) | **screen mode** (any BeamNG) |
|---|---|---|
| Frames | beamngpy `Camera` sensor, shared-memory stream | capture the game window (hood cam) |
| Telemetry | beamngpy `Electrics` (real speed) | none — model's own visual speed estimate |
| Control | direct `input.event` steering + pedals | a virtual Logitech G29 (uinput) |
| Map / vehicle | chosen for you, scripted or freeroam | you pick them in-game |

Tech mode is the high-fidelity path (the `Camera` sensor renders the
model's windshield view independently of what's on screen, so you can
film the AI car from any angle while it drives). Screen mode needs no
license — it captures whatever's on your monitor and drives a virtual
wheel — at the cost of a noisier, telemetry-free signal.

Inference runs on onnxruntime **ROCm** (AMD GPU) with CPU fallback.

## Status

- **M1 — model sees BeamNG** ✅ verified warp geometry (vx ratio 0.995,
  yaw 1.028, lane width 3.82 m); lane probs peak ~1.0 driving.
- **M2 — solid 20 Hz loop** ✅ shared-memory frame streaming
  (0.13 ms/frame); GPU pipeline ~6 ms/step on RDNA4 (gfx1201).
- **M3 — closed loop** ✅ lateral + end-to-end longitudinal, extensively
  tuned against real openpilot behavior (see *The control stack* and
  *What we learned*).
- **Screen mode** ✅ no-tech.key path: window capture + virtual G29,
  runs the full pipeline at 20 Hz with zero beamngpy.
- **UI** ✅ start panel and control panel rebuilt in dearpygui (crisp
  fonts, live camera texture, real plots, op-replay-clipper telemetry
  gauge).
- **M4 — scenarios** ⏳ traffic spawning works (start-panel slider);
  next: player-driven lead cut-ins, construction props, filming runs.

## Running

**Easiest path — the start panel:**

```bash
./start.sh          # (or: python tools/start_panel.py)
```

It auto-detects your BeamNG install and tech.key and reshapes itself:

- **tech.key present** → pick map, vehicle, and traffic. It launches
  the game with the tech server, waits for a real beamngpy handshake
  (not just the open port), then starts the control panel.
  `west_coast_usa` runs the scripted scenario at the calibrated spawn;
  any other map uses *freeroam interception* — load it in-game, enter a
  car, press START and the model hooks that car.
- **no tech.key** → screen mode. The map/vehicle options disappear
  (you set those in-game); you set the hood-cam FOV. Have BeamNG open
  with a car in **hood camera** view, press START, and the model
  captures the window and drives a virtual wheel.

Both spawns are detached, so closing the launcher (it auto-closes ~3 s
after START) never takes down the game or panel. The control panel logs
to `debug_out/control_panel_last.log`.

**Screen-mode wheel binding (one-time):** BeamNG applies its shipped
G29 inputmap automatically (steering = X axis, throttle = Y, brake =
RZ, both pedals inverted — the virtual wheel matches). If a pedal
doesn't respond, run `python -m simsteer.io.vwheel --sweep` and bind it
in Options → Controls.

**Manual pieces:**

```bash
bash launch_beamng.sh                       # host: BeamNG + tech server

python tools/control_panel.py               # tech mode, scripted west_coast_usa
python tools/control_panel.py --attach      # hook the car already in-game
python tools/control_panel.py --screen --fov 100     # screen mode (no beamngpy)
python tools/control_panel.py --model models/big_driving_supercombo.onnx
python tools/control_panel.py --split --vision <v.onnx> --policy <p.onnx>
python tools/control_panel.py --classic     # legacy hand-drawn cv2 UI
python tools/live_view.py --supercombo      # overlay viewer; drive manually
```

**Controls** (panel buttons or keys): `e` engage · `l` long-mode cycle
(EXP / CHILL / OFF) · `a`/`d` lane-change L/R · `z`/`c` turn L/R ·
`r` CAL · `v` camera-calib A/B · speed-cap slider or `-`/`=`.

## Requirements

Any Linux distro. Python 3.12 venv:

```bash
uv venv .venv --python 3.12        # or python3 -m venv .venv
uv pip install -r requirements.txt
```

CPU inference runs the full pipeline at ~60 Hz — the ROCm section is
optional. Screen mode additionally needs `evdev`, `mss`, `python-xlib`
(virtual wheel + window capture); the dearpygui UIs need `dearpygui`.

(Developed on Bazzite with the Python env in a distrobox, purely
because that's an easy place to put the ROCm libraries on an immutable
OS — a dev-machine detail, not a dependency. The game runs on the host;
only the control panel runs in the container.)

### Models (not in the repo)

Put the ONNX files in `models/`. Supercombo files may carry a release
suffix (e.g. `driving_supercombo_CD210.onnx`) — the default resolver
globs `driving_supercombo*.onnx`. The start panel toggles between a
single **supercombo** and a **split vision + policy** pair, and can
browse to any `.onnx`. Output layout is read from each file's own
metadata, so checkpoint swaps need no code changes.

- **Supercombo** (openpilot master): one `driving_supercombo*.onnx`.
- **Split pair** (openpilot 0.11.x): `driving_vision*.onnx` +
  `driving_policy*.onnx`.
- **Big model** (`big_driving_supercombo.onnx`, comma's USB-eGPU
  model, 1.76 GB): supported, and it's the only one with a real
  **action head** (a trained, smoothed accel/curvature output). It
  needs the game's graphics turned down enough to hold 20 Hz —
  otherwise its temporal buffers time-warp.

comma's LFS lives on GitLab (`gitlab.com/commaai/openpilot-lfs`), not
GitHub — pull pointers from `raw.githubusercontent`, then the GitLab
LFS `objects/batch` API.

### ROCm (optional, AMD GPUs)

`onnxruntime-rocm` from PyPI plus the ROCm 6.4 runtime libs
(`hipblas rocblas miopen-hip hip-runtime-amd hipfft hipsparse hiprand
rocrand rccl roctracer hipsolver rocsolver rocfft` from
repo.radeon.com), then `echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf
&& ldconfig`. Falls back to CPU automatically.

## The control stack

The model outputs a plan (trajectory + velocity/accel) and perception
(lane lines, road edges, leads). `simsteer/core/control.py` turns that
into steering and pedal commands, tuned to match how real openpilot
behaves.

**Lateral** — feed-forward from the plan's desired curvature through a
measured rack model, then executed at ~100 Hz:

- Rack fit (`axis = a·wheel`) is a **measured constant** from the CAL
  routine (a scripted steering system-ID), not an online learner — the
  sole exception is screen mode, which has no CAL and learns it live.
- **Understeer feed-forward**: the wheel target scales `(1 + kv·v²)`
  because tire slip grows with speed; fit from logged in-curve delivery
  (0.96 at city speed → 0.74 at highway).
- **100 Hz steering executor**: the model plans at 20 Hz but a separate
  thread actuates the wheel at ~100 Hz (openpilot's modeld/controlsd
  split), smoothing between plan updates instead of staircasing.
- **Rate limiter only** — the ISO curvature/lat-accel *magnitude*
  clamps are off by default (the sim's tires are the real limit); the
  ISO-jerk rate limiter (with faster unwind) shapes how fast the wheel
  turns.
- **Corner scanner** for entry speed: the plan's own upcoming curvature
  sets a comfort-bounded safe speed (`sqrt(a_lat_max/κ)`). The model
  plans its own turn speed; we no longer override it with a fixed
  intersection governor (that was compensating the speed under-read,
  now fixed at the source).

**Longitudinal** — follow the plan's velocity/accel through measured
pedal maps, with corrections that mirror the real model's envelope:

- Pedal maps are measured interpolation tables (throttle/brake vs
  commanded accel), since sim pedal response saturates and isn't affine.
- **Sim-scale speed correction** (the big one — see below): the plan
  velocity is scaled by the model's live `v_ego / pose_v` because a
  high camera makes the model under-read its speed. It's a per-vehicle
  **calibration** value now — persisted per car and warm-started, since
  it's driven by camera height (the bastion barely needs it, the tall
  pickup needs ~1.14).
- **E2E decel envelope**: off by default. It floored open-road decel to
  soften phantom hard-stops, but those came from the speed under-read
  (now fixed at the source), so it's redundant; the gated mechanism
  stays for a one-value re-enable if the model ever brakes too hard.
- Integral speed-trim, gated to steady cruise; a minimal openpilot-style
  stop-hold (brake ramp that releases when the plan wants speed).

**Best-effort principle:** like real openpilot, the controller does not
intervene on low vision confidence — it drives on whatever the model
outputs and the driver (you) is the fallback. There is no "lost-road"
safety stop.

## What we learned

The interesting findings, most impactful first — most were nailed by
decoding real comma **connect** route logs and comparing them to sim
runs frame-by-frame.

- **Traffic convention was inverted.** The model was being told it
  drove on the *left* (UK). On US maps that reads as "permanently in
  the oncoming lane" → left-lane bias, defensive phantom stops, launch
  hesitance. One-hot fix (`tc[int(is_rhd)]=1`, so US = `(1,0)`) turned
  20 mph timid crawling into 54 mph steady cruise. This was the single
  biggest behavior fix.
- **The speed under-read is camera height, not world scale.** A tall
  camera under-reads speed: ground optical flow scales as `v/h`, so the
  model (trained near ~1.2 m) reads a true 55 mph as ~48 from the
  pickup's 1.95 m mount (`pose_v/v_ego` ≈ 0.88), while the bastion at
  1.30 m reads it right (0.995). The *spatial* scale is fine — lane
  width 3.82 m vs 3.7 m, yaw 1.028 — so this is the only perception axis
  that's off, and it's **per-vehicle**. The model has no speed input, so
  in e2e it plans at its perceived-low speed and the controller brakes
  to a phantom stop (exactly why e2e never reached speed while chill
  mode did). Fixed by scaling the plan velocity by the model's own live
  `v_ego/pose_v`, now persisted as a per-vehicle calibration.
- **"Confidence" doesn't gate stopping.** `modelV2.confidence` is a
  RED/YELLOW/GREEN disengage-alert enum; on a smooth real drive it read
  "red" 95% of the time while never stopping, and lane confidence
  dropped to ~0 without stopping either. What keeps the real model
  smooth is a perfect speed estimate and a plan that never dips — not a
  confidence signal.
- **Camera resolution drives low-speed vision.** Bumping the render
  1664×832 → 2496×1248 (supersampling the model's warp like a real
  camera) raised lane confidence at 1-3 m/s from 0.26 to 0.75 and
  killed the "won't-commit crawl."
- **Steering timing, measured not guessed.** Command→response lag is
  ~0.30-0.35 s (cross-correlated from logs); the lead is matched to it,
  plus a small deliberate turn-in-early bias, which cured the "hugs the
  outside of curves then gets thrown across the lane" complaint.
- **The big model has an action head; CD210 doesn't.** The real car's
  smoothness partly comes from a trained, gently-bounded acceleration
  output. CD210's action slot is padding, so we derive accel from the
  plan, which is choppier — so if CD210 ever brakes too hard on open
  road, the (now-off) decel envelope is the intended remedy.
- **Gore points confuse E2E without navigation.** At road forks the
  model doesn't commit to a branch and drifts toward the dirt divider,
  then correctly slows because its path leaves the pavement. Real
  openpilot solves this with nav input (a bigger project); we mitigate
  by preferring continuous-highway routes.

## Vehicles

`beamng/world.py` (`VEHICLE_SPECS`) holds per-vehicle camera mount,
height, wheelbase, and part config, plus per-vehicle state files so one
vehicle's calibration never bleeds into another's.

- **bastion** — the measured-and-tuned sedan (camera ~1.30 m).
- **pickup** — Gavril D-Series crew-cab 4×4 automatic, a 2020 Sierra
  1500 AT4 recreation. Camera at the windshield glass, measured 1.95 m
  above the road.

## Upstream docs

simsteer's [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) still describes
the model core accurately (what's a comma port vs simsteer's own).
SETUP/TUNING are Windows-specific and don't apply here.

MIT, same as upstream. Models are comma.ai's.
