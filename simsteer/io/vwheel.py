"""Virtual steering wheel + pedals via /dev/uinput (evdev).

Presents itself as a Logitech G29 so BeamNG applies its bundled wheel
inputmap automatically — steering, throttle and brake Just Bind on
most installs; worst case the user does a one-time manual binding in
Options > Controls with `--sweep` wiggling the axes.

Axis conventions mirror the kernel hid-lg4ff driver so the ranges the
game expects line up:

    ABS_X   0..65535   steering, 32767 = center
    ABS_Z   0..255     throttle, 255 = RELEASED (inverted, like real hw)
    ABS_RZ  0..255     brake,    255 = RELEASED
    ABS_Y   0..255     clutch,   255 = RELEASED (never pressed by us)

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
_PEDAL_MAX = 255


class VirtualWheel:
    """A single always-on virtual wheel; call set() at any rate."""

    def __init__(self, name: str = "Logitech G29 Driving Force Racing Wheel"):
        caps = {
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=_STEER_MAX // 2, min=0,
                                  max=_STEER_MAX, fuzz=0, flat=0,
                                  resolution=0)),
                (e.ABS_Y, AbsInfo(value=_PEDAL_MAX, min=0, max=_PEDAL_MAX,
                                  fuzz=0, flat=0, resolution=0)),
                (e.ABS_Z, AbsInfo(value=_PEDAL_MAX, min=0, max=_PEDAL_MAX,
                                  fuzz=0, flat=0, resolution=0)),
                (e.ABS_RZ, AbsInfo(value=_PEDAL_MAX, min=0, max=_PEDAL_MAX,
                                   fuzz=0, flat=0, resolution=0)),
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
        t = int((1.0 - max(0.0, min(1.0, throttle or 0.0))) * _PEDAL_MAX)
        b = int((1.0 - max(0.0, min(1.0, brake or 0.0))) * _PEDAL_MAX)
        if (s, t, b) == self._last:
            return
        self._last = (s, t, b)
        self._ui.write(e.EV_ABS, e.ABS_X, s)
        self._ui.write(e.EV_ABS, e.ABS_Z, t)
        self._ui.write(e.EV_ABS, e.ABS_RZ, b)
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
