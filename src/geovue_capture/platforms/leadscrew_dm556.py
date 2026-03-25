"""
Platform configuration for the DM556 lead screw setup.

Hardware:
  - Driver:     DM556 stepper driver
  - PSU:        LRS 150-24 (24V, 6.5A)
  - Motor:      NEMA 23 (assumed)
  - Mechanism:  Lead screw linear stage
  - Controller: Raspberry Pi 5

DM556 DIP Switch Reference (8 switches):
  ┌──────────────────────────────────────────────────────────┐
  │ CURRENT SETTING (SW1-SW3)        SW4: Idle Current       │
  │ SW1  SW2  SW3  Peak Current(A)   ON  = 50% at idle       │
  │ ON   ON   ON   1.0               OFF = full current      │
  │ OFF  ON   ON   1.5                                       │
  │ ON   OFF  ON   2.0                                       │
  │ OFF  OFF  ON   2.5                                       │
  │ ON   ON   OFF  2.8                                       │
  │ OFF  ON   OFF  3.2                                       │
  │ ON   OFF  OFF  4.0                                       │
  │ OFF  OFF  OFF  4.5                                       │
  ├──────────────────────────────────────────────────────────┤
  │ MICROSTEP SETTING (SW5-SW8)                              │
  │ SW5  SW6  SW7  SW8  Pulses/Rev  Microstep                │
  │ ON   ON   ON   ON   200         Full step                │
  │ OFF  ON   ON   ON   400         1/2  step                │
  │ ON   OFF  ON   ON   800         1/4  step                │
  │ OFF  OFF  ON   ON   1600        1/8  step   ← WORKING   │
  │ ON   ON   OFF  ON   3200        1/16 step                │
  │ OFF  ON   OFF  ON   6400        1/32 step                │
  │ ON   OFF  OFF  ON   12800       1/64 step                │
  │ OFF  OFF  OFF  ON   25600       1/128 step  ← BARELY     │
  │                                               MOVES!     │
  └──────────────────────────────────────────────────────────┘

  IMPORTANT: Check YOUR specific DM556 datasheet - some manufacturers
  swap current and microstep switch groups or use different tables.
  The table above is for the most common variant (StepperOnline/Leadshine).

  Known-good setting: OFF,ON,ON,OFF,OFF,OFF,ON,ON
    → Current: 1.5A peak, no idle reduction
    → Microstep: 1600 pulses/rev (1/8 step)

GPIO Wiring (BCM numbering):
  Pin 11 (GPIO 17) → PUL+ (step pulses)
  Pin 13 (GPIO 27) → DIR+ (direction)
  Pin 15 (GPIO 22) → ENA+ (enable, active LOW)
  Pin 9  (GND)     → PUL-, DIR-, ENA- (common ground)

Limit Switches:
  Switch 1 (home):  GPIO 13 (output/ground) ↔ GPIO 19 (input, pull-up)
  Switch 2 (far):   GPIO 26 (input, pull-up) ↔ Pin 39 (GND)
"""

from pathlib import Path
from .base import (
    PlatformConfig,
    StepperPins,
    LimitSwitchPins,
    MotionProfile,
    StepperDriverConfig,
)


class LeadscrewDM556Config(PlatformConfig):
    """Configuration for the DM556 + lead screw linear stage."""

    def __init__(self):
        super().__init__(
            platform_name="leadscrew",
            platform_description="DM556 driver + LRS 150-24 PSU + lead screw stage",

            # GPIO - matches existing wiring
            stepper_pins=StepperPins(
                step=17,       # GPIO 17 (Pin 11)
                direction=27,  # GPIO 27 (Pin 13)
                enable=22,     # GPIO 22 (Pin 15)
            ),
            limit_switch_pins=LimitSwitchPins(
                home=19,       # GPIO 19 (Pin 35) - switch 1 input
                far=26,        # GPIO 26 (Pin 37) - switch 2 input
                ground=13,     # GPIO 13 (Pin 33) - acts as ground for switch 1
            ),

            # DM556 driver config
            # Known-good DIP setting: OFF,ON,ON,OFF,OFF,OFF,ON,ON
            #   → 1.5A peak, 1600 pulses/rev
            driver=StepperDriverConfig(
                driver_model="DM556",
                microsteps=1600,       # 1/8 step → 1600 pulses/rev
                current_ma=1500,       # 1.5A peak (SW1-SW3: OFF,ON,ON)
                idle_current_pct=100,  # SW4=OFF → no idle reduction
                step_pulse_us=2.5,     # DM556 min pulse width
                dir_setup_us=5.0,      # Direction setup time
                enable_active_low=True,
            ),

            # Lead screw motion profile
            # Lead screw pitch ~2mm/rev (typical T8), 1600 steps/rev → 800 steps/mm
            # Adjust steps_per_mm based on YOUR actual lead screw pitch
            motion=MotionProfile(
                steps_per_mm=800.0,     # 1600 microsteps / 2mm pitch
                max_speed_mm_s=30.0,    # Conservative for lead screw
                acceleration_mm_s2=100.0,
                home_speed_mm_s=8.0,
                travel_length_mm=600.0, # ~60cm usable travel
                backlash_comp_mm=0.1,   # Lead screw backlash
            ),

            # This machine has no touchscreen
            has_touchscreen=False,
            window_width=1024,
            window_height=768,

            # Stage server
            stage_server_port=5055,

            # USB hub path for camera power control
            camera_usb_hub_path="1-1",
        )


# Quick reference for adjusting DIP switches
DM556_DIP_PRESETS = {
    "working_1600": {
        "switches": "OFF,ON,ON,OFF,OFF,OFF,ON,ON",
        "current_a": 1.5,
        "microsteps": 1600,
        "notes": "Known-good setting for this machine",
    },
    "higher_current_1600": {
        "switches": "OFF,OFF,ON,OFF,OFF,OFF,ON,ON",
        "current_a": 2.5,
        "microsteps": 1600,
        "notes": "More torque, same resolution",
    },
    "smooth_3200": {
        "switches": "OFF,ON,ON,OFF,ON,ON,OFF,ON",
        "current_a": 1.5,
        "microsteps": 3200,
        "notes": "Smoother motion, need to double pulse rate",
    },
}
