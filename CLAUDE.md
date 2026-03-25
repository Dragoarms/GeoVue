# GeoVue Capture 2.8 - Development Instructions

## Project Overview

GeoVue Capture is a Raspberry Pi 5 application for automated chip tray photography in geological drilling operations. It controls a motorized linear stage to position chip trays under a camera for capture.

**Version: 2.8.0**

## Two Hardware Platforms

### 1. Leadscrew (DM556) - "this machine"
- **Driver:** DM556 stepper driver
- **PSU:** LRS 150-24 (24V, 6.5A)
- **Mechanism:** Lead screw linear stage
- **Display:** Standard HDMI monitor (no touchscreen)
- **Controller:** Raspberry Pi 5

**DM556 DIP Switch - Known-good setting:** `OFF,ON,ON,OFF,OFF,OFF,ON,ON`
- SW1-SW3: Current = 1.5A peak
- SW4: Idle current reduction OFF
- SW5-SW8: Microstep = 1600 pulses/rev (1/8 step)

**GPIO Wiring (BCM):**
- GPIO 17 (Pin 11) → PUL+ (step pulses)
- GPIO 27 (Pin 13) → DIR+ (direction)
- GPIO 22 (Pin 15) → ENA+ (enable, active LOW)
- Pin 9 (GND) → PUL-, DIR-, ENA-
- Limit Switch 1 (home): GPIO 13 (output/ground) ↔ GPIO 19 (input, pull-up)
- Limit Switch 2 (far): GPIO 26 (input, pull-up) ↔ Pin 39 (GND)

### 2. Pilbara (TMC2225)
- **Driver:** TMC2225 (or compatible TMC silent stepper)
- **Mechanism:** GT2 timing belt linear stage
- **Display:** Touchscreen
- **Controller:** Raspberry Pi 5

**GPIO Wiring (BCM) - may differ, check actual wiring:**
- GPIO 12 → STEP (hardware PWM capable)
- GPIO 13 → DIR
- GPIO 6 → EN
- GPIO 19 → Home limit switch
- GPIO 26 → Far limit switch

## Existing Code on This Pi

There is legacy/reference code in these local directories (NOT in git):
- `/home/geovue/Documents/PWM RPI 5 GeoVueCapture GUI/` - Leadscrew version GUI
- `/home/geovue/Documents/GeoVue_Capture_pi/Final Pilbara/` - Pilbara version with touchscreen GUI and limit switch logic

**IMPORTANT:** Read and examine the "Final Pilbara" code carefully before building the GUI. It contains:
- The desired GUI layout and look/feel
- How limit switches must be respected at ALL times (always-on safety)
- Touchscreen interaction patterns

## What Needs To Be Done

### Completed (in `src/geovue_capture/`):
- [x] Platform configs (`platforms/`) - DM556 leadscrew + TMC2225 Pilbara
- [x] Pi 5 GPIO abstraction (`hardware/gpio_manager.py`) - gpiod/lgpio/mock backends
- [x] Limit switch monitor (`hardware/limit_switches.py`) - always-on, blocks unsafe moves
- [x] Stepper controller (`hardware/stepper.py`) - checks limits before EVERY step
- [x] ArUco detector (`imaging/aruco_detector.py`)
- [x] Data/session manager (`imaging/data_manager.py`)
- [x] Bilingual strings (`gui/i18n.py`) - French and English

### Still TODO:
- [ ] **Main GUI** (`gui/capture_app.py`) - Tkinter GUI based on the Final Pilbara code's look/feel
  - Must be fully bilingual (French + English) using `gui/i18n.py`
  - Language toggle in the UI
  - Touchscreen-friendly layout when `config.has_touchscreen == True`
  - Always-visible limit switch status indicators
  - Always-visible emergency stop button
  - Stage control panel (left/right jog, home, stop)
  - Data entry (Hole ID, depth range, moisture)
  - Processed-today list
  - NO camera/phone/ADB functionality (skip all that)
- [ ] **Entry point** (`main.py`) - Platform detection, arg parsing, launch GUI
- [ ] **Pi setup script** (`setup/pi_setup.sh`) - See requirements below
- [ ] **pyproject.toml** - Version 2.8.0, Pi dependencies
- [ ] Integrate and test on actual hardware

### Pi Setup Script Requirements (`setup/pi_setup.sh`):
The setup script must:
1. **Auto power-on:** Configure the Pi to boot when it receives power (set `wake_on_gpio` or equivalent in bootloader config)
2. **Kiosk mode:** Set up the GeoVue Capture GUI to run on startup in kiosk mode
   - Fullscreen, no window decorations, no taskbar access
   - Can ONLY exit kiosk mode with password `6218`
3. **PWM pin setup:** Ensure hardware PWM pins are correctly configured in `/boot/firmware/config.txt` (dtoverlay for PWM)
4. **GPIO permissions:** Add user to gpio group, set up udev rules
5. **Dependencies:** Install system packages (python3, python3-gpiod, python3-lgpio, python3-tk, python3-pil, python3-opencv, etc.)
6. **Systemd service:** Create a service file that launches the capture app on boot
7. **Auto-login:** Configure auto-login to desktop for the geovue user
8. **Disable screen blanking:** Keep display always on during operation

## Key Design Principles

1. **Limit switches are SACRED** - The monitor thread runs at 1ms polling. Every single step pulse checks `is_move_allowed()` BEFORE firing. When a switch triggers, the motor stops instantly and can only move AWAY from the triggered switch.

2. **No camera/phone code** - Skip all ADB, phone camera, USB power cycling, screenshot stuff. The capture workflow will be added later.

3. **Bilingual always** - Every user-facing string goes through `gui/i18n.py`. Use `t("key")` everywhere, never hardcode English or French strings in the GUI code.

4. **Pi 5 compatible** - Use gpiod or lgpio, NOT RPi.GPIO (which doesn't work on Pi 5).

## Running the App

```bash
# From repo root
sudo python3 -m geovue_capture --platform leadscrew  # or --platform pilbara
```

## File Structure
```
src/geovue_capture/
├── __init__.py              # Version 2.8.0
├── main.py                  # Entry point (TODO)
├── platforms/
│   ├── __init__.py          # Platform detection + registry
│   ├── base.py              # Base config dataclasses
│   ├── leadscrew_dm556.py   # DM556 + lead screw config
│   └── pilbara_tmc2225.py   # TMC2225 + GT2 belt config
├── hardware/
│   ├── __init__.py
│   ├── gpio_manager.py      # Pi 5 GPIO abstraction
│   ├── limit_switches.py    # Always-on limit switch monitor
│   └── stepper.py           # Stepper controller with safety
├── imaging/
│   ├── __init__.py
│   ├── aruco_detector.py    # ArUco marker detection
│   └── data_manager.py      # Session + image file management
├── gui/
│   ├── __init__.py
│   ├── i18n.py              # French/English translations
│   └── capture_app.py       # Main GUI (TODO)
└── setup/
    └── pi_setup.sh          # Pi deployment script (TODO)
```
