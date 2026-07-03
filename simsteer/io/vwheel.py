"""Virtual steering wheel + pedals via /dev/uinput (evdev).

Presents itself as a Logitech G29 so BeamNG applies its bundled wheel
inputmap automatically — steering, throttle and brake Just Bind on
most installs; worst case the user does a one-time manual binding in
Options > Controls with `--sweep` wiggling the axes.

BeamNG reads this as a RAW joystick (its input screen shows generic
xaxis/zaxis/rxaxis, not a named G29 inputmap), so the pedals are
PLAIN and non-inverted — 0 at rest, max when pressed — instead of the
real-hardware "rest = max, inputmap inverts it back" convention. With
raw axes that convention read as full-throttle-at-rest. Bind in-game
Options > Controls (steering = xaxis, throttle = zaxis, brake =
rxaxis), no axis-invert needed.

Axis assignment matches what BeamNG's binding screen actually reads
off this device (verified in-game): steering = X, throttle = Y,
brake = RZ ("R Z AXIS"). The earlier throttle-on-Z was dead because
BeamNG binds throttle to the Y axis, which we were leaving at zero.

    ABS_X   0..65535   steering, 32767 = center
    ABS_Y   0..65535   throttle, 0 = released, 65535 = full  (BeamNG "Y")
    ABS_RZ  0..65535   brake,    0 = released, 65535 = full  (BeamNG "R Z")
    ABS_Z   0..65535   spare, always 0 (kept so SDL axis indices are stable)

Permissions: /dev/uinput must be writable. On this dev machine the
seat ACL already grants it; elsewhere add the udev rule printed by
the error message and re-plug into the `input` group.

Test:  .venv/bin/python3 -m simsteer.io.vwheel --sweep
"""
from __future__ import annotations

import time

try:
    from evdev import AbsInfo, UInput, ecodes as e
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        "python-evdev is required for the virtual wheel: "
        "uv pip install evdev") from exc

_PERM_HELP = """\
could not create the uinput device: {err}

/dev/uinput isn't writable. Fix (once):
  echo 'KERNEL=="uinput", GROUP="input", MODE="0660", TAG+="uaccess"' \\
    | sudo tee /etc/udev/rules.d/99-uinput.rules
  sudo udevadm control --reload && sudo modprobe uinput
then log out/in (or: sudo usermod -aG input $USER)."""

_STEER_MAX = 65535
_PEDAL_MAX = 65535


class VirtualWheel:
    """A single always-on virtual wheel; call set() at any rate."""

    def __init__(self, name: str = "Logitech G29 Driving Force Racing Wheel"):
        # Declare the FULL 6-axis set (X Y Z RX RY RZ). BeamNG names
        # joystick axes by POSITION (x,y,z,rx,ry,rz), so its default
        # G29 bindings — throttle="y" (pos 2), brake="rz" (pos 6) — only
        # line up if ABS_RZ actually sits in the 6th slot. With only 4
        # axes ABS_RZ fell into the "rx" slot (pos 4) and the brake
        # binding pointed at a slot we never drove. The RX/RY spares
        # push ABS_RZ to pos 6 = "rz" where the brake is bound.
        _p = dict(min=0, max=_PEDAL_MAX, fuzz=0, flat=0, resolution=0)
        caps = {
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=_STEER_MAX // 2, min=0,
                                  max=_STEER_MAX, fuzz=0, flat=0,
                                  resolution=0)),
                (e.ABS_Y, AbsInfo(value=0, **_p)),    # pos 2 "y"  throttle
                (e.ABS_Z, AbsInfo(value=0, **_p)),    # pos 3 "z"  spare
                (e.ABS_RX, AbsInfo(value=0, **_p)),   # pos 4 "rx" spare
                (e.ABS_RY, AbsInfo(value=0, **_p)),   # pos 5 "ry" spare
                (e.ABS_RZ, AbsInfo(value=0, **_p)),   # pos 6 "rz" brake
            ],
            # a few bindable buttons (engage toggles etc. if ever wanted)
            e.EV_KEY: [e.BTN_TRIGGER, e.BTN_THUMB, e.BTN_TOP, e.BTN_TOP2],
        }
        try:
            self._ui = UInput(caps, name=name,
                              vendor=0x046D, product=0xC24F, version=0x0111)
        except (PermissionError, OSError) as exc:
            raise RuntimeError(_PERM_HELP.format(err=exc)) from exc
        self._last = (None, None, None)
        # give the game's device scanner a beat to enumerate us
        time.sleep(0.2)

    # ---- control ----

    def set(self, steer: float, throttle: float | None,
            brake: float | None) -> None:
        """steer -1..1 (right positive, matching the model/axis frame);
        throttle/brake 0..1, None = release that pedal."""
        s = int((max(-1.0, min(1.0, steer)) * 0.5 + 0.5) * _STEER_MAX)
        # non-inverted: 0 at rest, max when pressed (see module docstring)
        t = int(max(0.0, min(1.0, throttle or 0.0)) * _PEDAL_MAX)
        b = int(max(0.0, min(1.0, brake or 0.0)) * _PEDAL_MAX)
        if (s, t, b) == self._last:
            return
        self._last = (s, t, b)
        self._ui.write(e.EV_ABS, e.ABS_X, s)
        self._ui.write(e.EV_ABS, e.ABS_Y, t)     # BeamNG "Y AXIS" = throttle
        self._ui.write(e.EV_ABS, e.ABS_RZ, b)    # BeamNG "R Z AXIS" = brake
        self._ui.syn()

    def neutral(self) -> None:
        self.set(0.0, 0.0, 0.0)

    def close(self) -> None:
        try:
            self.neutral()
            self._ui.close()
        except OSError:
            pass


def _sweep() -> None:                            # pragma: no cover
    """Wiggle each axis so the game's binding screen can see them."""
    import math
    vw = VirtualWheel()
    print("virtual wheel up — open the game's controls/bindings screen.")
    print("sweeping: 10 s steering, then throttle, then brake. Ctrl-C stops.")
    try:
        t0 = time.time()
        while True:
            t = time.time() - t0
            phase = int(t // 10) % 3
            w = math.sin(t * 2.0)
            vw.set(w if phase == 0 else 0.0,
                   max(0.0, w) if phase == 1 else 0.0,
                   max(0.0, w) if phase == 2 else 0.0)
            print(f"\r{['STEER', 'THROTTLE', 'BRAKE'][phase]:9s} "
                  f"{w:+.2f}   ", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        vw.close()
        print("\nwheel closed.")


if __name__ == "__main__":                       # pragma: no cover
    import sys
    if "--sweep" in sys.argv:
        _sweep()
    else:
        vw = VirtualWheel()
        print("created OK; --sweep to exercise axes")
        vw.close()
