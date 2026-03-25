"""
Platform configuration for the Pilbara field unit.

Hardware:
  - Driver:     TMC2225 (or compatible TMC stepper driver)
  - Mechanism:  GT2 timing belt linear stage
  - Display:    Touchscreen
  - Controller: Raspberry Pi 5

TMC2225 Notes:
  - Silent stepper driver with StealthChop
  - Configured via UART or standalone (MS1/MS2 pins)
  - Default 256 microstep interpolation from any base microstep
  - Standalone mode: MS1/MS2 set base microstep resolution
  - Typical Vref sets current limit

GT2 Belt Drive:
  - GT2 belt pitch: 2mm
  - Typical pulley: 20 teeth → 40mm per revolution
  - Much faster than lead screw, no backlash
  - Higher acceleration possible

GPIO Wiring (BCM numbering) - ADJUST TO MATCH YOUR WIRING:
  GPIO 17 → STEP (step pulses)
  GPIO 27 → DIR  (direction)
  GPIO 22 → EN   (enable, active LOW)

  Alternative GPIO if different from leadscrew version:
  GPIO 12 → STEP (can use hardware PWM on Pi 5)
  GPIO 13 → DIR
  GPIO 6  → EN

Limit Switches:
  GPIO 19 → Home switch (input, pull-up)
  GPIO 26 → Far switch  (input, pull-up)
"""

from pathlib import Path
from .base import (
    PlatformConfig,
    StepperPins,
    LimitSwitchPins,
    MotionProfile,
    StepperDriverConfig,
)


class PilbaraTMC2225Config(PlatformConfig):
    """Configuration for the Pilbara field unit with TMC2225 + GT2 belt."""

    def __init__(self):
        super().__init__(
            platform_name="pilbara",
            platform_description="TMC2225 driver + GT2 timing belt + touchscreen",

            # GPIO pins - adjust these to match Pilbara wiring
            # Using alternative pins to avoid conflicts with touchscreen
            stepper_pins=StepperPins(
                step=12,       # GPIO 12 - supports hardware PWM
                direction=13,  # GPIO 13
                enable=6,      # GPIO 6
            ),
            limit_switch_pins=LimitSwitchPins(
                home=19,       # GPIO 19
                far=26,        # GPIO 26
                ground=None,   # No GPIO-as-ground needed (use actual GND)
            ),

            # TMC2225 driver config
            driver=StepperDriverConfig(
                driver_model="TMC2225",
                microsteps=800,        # 1/4 step base (TMC interpolates to 256)
                current_ma=1200,       # TMC2225 max ~1.4A, run at 1.2A
                idle_current_pct=50,   # StealthChop handles idle nicely
                step_pulse_us=1.0,     # TMC2225 is fast
                dir_setup_us=2.0,
                enable_active_low=True,
            ),

            # GT2 belt motion profile
            # 20T pulley: 20 teeth × 2mm pitch = 40mm/rev
            # 800 microsteps/rev → 800/40 = 20 steps/mm
            motion=MotionProfile(
                steps_per_mm=20.0,       # 800 microsteps / 40mm per rev
                max_speed_mm_s=100.0,    # Belt drives are fast
                acceleration_mm_s2=500.0,
                home_speed_mm_s=20.0,
                travel_length_mm=700.0,  # Adjust for actual Pilbara stage
                backlash_comp_mm=0.0,    # Belt drives have no backlash
            ),

            # Pilbara has a touchscreen
            has_touchscreen=True,
            window_width=800,   # Touchscreen resolution
            window_height=480,

            # Stage server
            stage_server_port=5055,

            # USB hub path - may differ on Pilbara
            camera_usb_hub_path="1-1",
        )
