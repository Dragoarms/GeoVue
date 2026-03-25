"""Base platform configuration - all platforms inherit from this."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StepperPins:
    """GPIO pin assignments for stepper motor control (BCM numbering)."""
    step: int       # PUL/STEP pulse pin
    direction: int  # DIR direction pin
    enable: int     # ENA enable pin


@dataclass
class LimitSwitchPins:
    """GPIO pin assignments for limit switches (BCM numbering)."""
    home: int       # Home-end limit switch input
    far: int        # Far-end limit switch input
    ground: Optional[int] = None  # GPIO used as ground source (if needed)


@dataclass
class MotionProfile:
    """Motion parameters for the linear stage."""
    steps_per_mm: float          # Steps (after microstepping) per mm of travel
    max_speed_mm_s: float        # Maximum travel speed in mm/s
    acceleration_mm_s2: float    # Acceleration in mm/s²
    home_speed_mm_s: float       # Speed when homing
    travel_length_mm: float      # Total usable travel distance
    backlash_comp_mm: float = 0  # Backlash compensation


@dataclass
class StepperDriverConfig:
    """Stepper driver configuration."""
    driver_model: str
    microsteps: int              # Microstep resolution (e.g. 1600 pulses/rev)
    current_ma: int              # Peak current in milliamps
    idle_current_pct: int = 50   # Idle current as % of peak
    step_pulse_us: float = 5.0   # Minimum step pulse width in microseconds
    dir_setup_us: float = 5.0    # Direction setup time in microseconds
    enable_active_low: bool = True  # Most drivers: LOW = enabled


@dataclass
class PlatformConfig:
    """Complete platform configuration. Subclasses set actual values."""

    # Identity
    platform_name: str = "base"
    platform_description: str = ""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path.home() / "GeoVue")
    storage_dir: Path = field(default_factory=lambda: Path.home() / "GeoVue" / "storage")

    # GPIO
    stepper_pins: StepperPins = field(default_factory=lambda: StepperPins(17, 27, 22))
    limit_switch_pins: LimitSwitchPins = field(default_factory=lambda: LimitSwitchPins(19, 26))

    # Driver
    driver: StepperDriverConfig = field(default_factory=lambda: StepperDriverConfig(
        driver_model="generic",
        microsteps=1600,
        current_ma=2000,
    ))

    # Motion
    motion: MotionProfile = field(default_factory=lambda: MotionProfile(
        steps_per_mm=80,
        max_speed_mm_s=50,
        acceleration_mm_s2=200,
        home_speed_mm_s=10,
        travel_length_mm=600,
    ))

    # Camera
    camera_usb_hub_path: str = "1-1"

    # Stage server
    stage_server_port: int = 5055

    # UI
    has_touchscreen: bool = False
    window_width: int = 1024
    window_height: int = 768

    @property
    def images_dir(self):
        return self.storage_dir / "Images to Process"

    @property
    def deleted_dir(self):
        return self.storage_dir / "Deleted files"

    @property
    def temp_dir(self):
        return self.storage_dir / "temp"

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        for d in [self.storage_dir, self.images_dir, self.deleted_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)
