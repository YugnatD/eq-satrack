# eq-satrack

Real-time satellite tracking for a ZWO AM3/AM5/AM7 equatorial mount and a
ZWO ASI-series planetary camera. Built to catch the ISS (and, via NORAD ID,
any other satellite with a published TLE) crossing the sky in the ~30-90
seconds a low pass gives you, with a live-adjustable feedforward tracking
loop, camera-based closed-loop auto-guiding, and a GUI built for actually
standing at the mount with a laptop mid-pass.

The mount side talks the AM5's serial protocol directly (LX200-derived, see
`docs/`) rather than going through INDI/ASCOM, and everything is developed
and testable against a mock mount/camera before it ever touches real
hardware — see [Development against a mock](#development-against-a-mock).

## Features

- **Pass prediction** — fetches a live TLE (Celestrak) for the ISS or any
  custom NORAD ID, computes rise/set/culmination and meridian crossings for
  your site, shown on a sky-track polar plot.
- **Feedforward tracking loop** — commands `:Rv#`/`:Me#`/`:Mn#` at 20 Hz
  from the precomputed trajectory, with a live along-track (delta_t) and
  cross-track (perpendicular nudge) manual trim, an optional empirically-fed
  `mount_lag_s` feedforward lead time, and an opt-in PI feedback trim —
  along-track error settles to single-digit arcseconds against real ISS
  passes on an AM3 (see the tests and code comments for the real-hardware
  numbers this was tuned against).
- **Automatic pier-side correction** — a German equatorial mount's DEC axis
  physically reverses sense on a pier flip; `AxisSigns` tracks which side a
  calibration is valid for and re-corrects itself the moment the mount
  reports a different side, instead of silently commanding DEC backwards.
- **Camera-based auto-guiding** — detects the satellite as the brightest
  blob in the live preview and feeds a cross-track correction into the same
  offset a human operator would apply by hand; a short nudge-based
  calibration step maps pixels to arcseconds for whatever rotation/mirroring
  your optical train happens to have.
- **SER/FITS capture** — records the raw video stream to a SER file with a
  hand-verified binary header, plus single-frame FITS snapshots.
- **Safety net** — a watchdog thread issues an emergency stop if commands
  stop arriving while the mount is supposed to be moving; a big always-live
  EMERGENCY STOP button; typed-confirmation gates before any real
  fast/tube-attached move; controls that touch the mount grey out while
  parked or disconnected instead of silently no-op'ing.
- **Realistic mock mode** — `MockMount` simulates serial latency, a
  first-order motor response ramp, and real pier-side geometry; the mock
  camera renders a synthetic star field (from a real Hipparcos extract) with
  a moving ISS blob. The whole GUI, and every test, runs against these with
  no hardware attached.

## Hardware this targets

- ZWO AM3 / AM5 / AM7 equatorial mount (serial, 9600 baud, or over TCP)
- ZWO ASI-series camera (developed against an ASI290MC) — optional, the
  mount side works standalone
- Everything also runs fully mocked with no hardware at all

## Setup

```bash
git clone <this repo>
cd eq-satrack
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Real camera support additionally needs the ZWO ASI SDK
(`libASICamera2.so`) installed and discoverable — the GUI and camera mock
work fine without it, this is only needed to talk to real ASI hardware.

### Star catalog

`assets/bright_stars.npz` (a filtered Hipparcos extract, ~1.1 MB) is
committed and is all the mock camera needs — nothing to do. If you want to
regenerate it from the raw catalog yourself, `skyfield`'s loader will fetch
`hip_main.dat` (~50 MB) on first use of the Hipparcos loading path.

## Quick start

```bash
# GUI, defaults to picking mock/serial/TCP from the Connection tab
python3 gui.py

# CLI, fully mocked, no hardware needed
python3 track_pass.py --mock --skip-confirm

# CLI against real hardware on /dev/ttyACM0
python3 track_pass.py --serial /dev/ttyACM0
```

`characterize.py` is the lower-level probing tool this project's
understanding of the wire protocol was built from — see its own `--help`.

## Project layout

```
am5/            Mount protocol, tracking loop, ephemeris, GUI
  gui/          tkinter app (Connection / Passes / Calibration /
                Exposure calc / Transit tabs, floating jog window)
camera/         ASI camera wrapper + mock, SER/FITS writers, auto-guiding
tests/          pytest suite — runs entirely against mocks, no hardware
docs/           ZWO's own serial protocol reference PDF
assets/         Star catalog extract, ISS reference photo (both public
                domain / freely redistributable — see their .LICENSE.txt)
characterize.py Low-level protocol probing/verification script
track_pass.py   CLI: predict the next pass and track it, no GUI
gui.py          GUI entry point
```

## Development against a mock

Every device-touching path has a mock counterpart (`MockMount`,
`MockAsiCamera`) built to match real hardware behavior as closely as
possible — including quirks discovered the hard way, like simulated serial
latency and a first-order motor response ramp. New mount/camera features
in this project are built and tested against the mock first, then verified
against real hardware before being considered done. If you're extending
this without the hardware in front of you, that's the intended workflow:

```bash
pytest                          # full suite, no hardware needed
python3 gui.py                  # then pick "Mock" on the Connection tab
python3 track_pass.py --mock --skip-confirm
```

## Safety

This drives real motors. A few things are load-bearing, not incidental:

- The tube/OTA must be off the mount for anything that slews at high jog
  rates — `characterize.py`/`track_pass.py` gate this behind a typed
  `TUBE REMOVED` confirmation.
- A real tracking pass (OTA attached) gates behind a separate typed
  `READY TO TRACK` confirmation checking pier side / cable slack / starting
  position.
- The watchdog (`am5/safety.py`) auto-stops if commands stop arriving
  while the mount was told to keep moving — but it runs in the same
  process as everything else, so it is not a substitute for being at the
  mount with a hand on the power switch during real hardware testing.
- The EMERGENCY STOP button in the GUI is never gated by connection state,
  parked state, or anything else.

## License

MIT — see [LICENSE](LICENSE). Third-party assets in `assets/` carry their
own attribution in the matching `.LICENSE.txt` (public domain ESA
Hipparcos data and a public domain NASA photo). `docs/`'s protocol PDF is
ZWO's own vendor documentation, included for reference.
