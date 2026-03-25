"""
Stepper motor controller with integrated limit switch safety.

Every single step pulse checks limit switches BEFORE firing.
Supports:
  - Continuous movement (jog) in either direction
  - Precise step counts
  - Trapezoidal acceleration profiles
  - Homing to limit switch
  - Emergency stop at any time

This works with any step/direction stepper driver (DM556, TMC2225, etc).
"""

import logging
import math
import threading
import time
from enum import Enum, auto
from typing import Optional

from ..platforms.base import PlatformConfig
from .gpio_manager import GPIOBackend, HIGH, LOW
from .limit_switches import LimitSwitchMonitor

logger = logging.getLogger(__name__)


class MotorState(Enum):
    IDLE = auto()
    MOVING = auto()
    HOMING = auto()
    ERROR = auto()


class StepperController:
    """
    High-level stepper motor controller.

    All movement respects limit switches at ALL times.
    """

    def __init__(self, config: PlatformConfig, gpio: GPIOBackend,
                 limit_monitor: LimitSwitchMonitor):
        self._config = config
        self._gpio = gpio
        self._limits = limit_monitor

        self._pins = config.stepper_pins
        self._driver = config.driver
        self._motion = config.motion

        # State
        self._state = MotorState.IDLE
        self._position_steps: int = 0  # Current position in steps (0 = home)
        self._direction_home: bool = True  # True = toward home
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._move_thread: Optional[threading.Thread] = None

        # Setup GPIO pins
        self._setup_pins()

    def _setup_pins(self):
        """Configure stepper motor GPIO pins."""
        self._gpio.setup_output(self._pins.step, LOW)
        self._gpio.setup_output(self._pins.direction, LOW)
        self._gpio.setup_output(self._pins.enable, HIGH if self._driver.enable_active_low else LOW)

        logger.info(
            f"Stepper pins: STEP=GPIO{self._pins.step}, "
            f"DIR=GPIO{self._pins.direction}, ENA=GPIO{self._pins.enable}"
        )

    def enable(self):
        """Enable the stepper driver (motor holds position)."""
        value = LOW if self._driver.enable_active_low else HIGH
        self._gpio.write(self._pins.enable, value)
        logger.debug("Motor enabled")

    def disable(self):
        """Disable the stepper driver (motor free to rotate)."""
        value = HIGH if self._driver.enable_active_low else LOW
        self._gpio.write(self._pins.enable, value)
        logger.debug("Motor disabled")

    def _set_direction(self, toward_home: bool):
        """Set direction pin and wait for setup time."""
        self._direction_home = toward_home
        # toward_home = LOW, away_from_home = HIGH (adjust if your wiring differs)
        self._gpio.write(self._pins.direction, LOW if toward_home else HIGH)
        time.sleep(self._driver.dir_setup_us / 1_000_000)

    def _single_step(self) -> bool:
        """
        Execute a single step pulse.

        Returns True if step was executed, False if blocked by limit switch.
        ALWAYS checks limit switches before pulsing.
        """
        # SAFETY: Check limit switch BEFORE every step
        if not self._limits.is_move_allowed(self._direction_home):
            return False

        if self._stop_event.is_set():
            return False

        # Generate step pulse
        pulse_s = self._driver.step_pulse_us / 1_000_000
        self._gpio.write(self._pins.step, HIGH)
        time.sleep(pulse_s)
        self._gpio.write(self._pins.step, LOW)
        time.sleep(pulse_s)

        # Update position tracking
        if self._direction_home:
            self._position_steps -= 1
        else:
            self._position_steps += 1

        return True

    def stop(self):
        """Stop all movement immediately."""
        self._stop_event.set()
        if self._move_thread and self._move_thread.is_alive():
            self._move_thread.join(timeout=2.0)
        self._state = MotorState.IDLE
        logger.info("Motor stopped")

    def move_steps(self, steps: int, speed_steps_per_sec: Optional[float] = None,
                   accelerate: bool = True, blocking: bool = True):
        """
        Move a specific number of steps.

        Args:
            steps: Positive = away from home, negative = toward home.
            speed_steps_per_sec: Override speed. None = use max from config.
            accelerate: If True, use trapezoidal acceleration profile.
            blocking: If True, wait for move to complete.
        """
        if steps == 0:
            return

        self._stop_event.clear()
        toward_home = steps < 0
        total_steps = abs(steps)

        if speed_steps_per_sec is None:
            speed_steps_per_sec = self._motion.max_speed_mm_s * self._motion.steps_per_mm

        def _run():
            self.enable()
            self._set_direction(toward_home)
            self._state = MotorState.MOVING

            if accelerate:
                self._move_with_accel(total_steps, speed_steps_per_sec)
            else:
                step_delay = 1.0 / speed_steps_per_sec
                for _ in range(total_steps):
                    if not self._single_step():
                        break
                    time.sleep(step_delay)

            self._state = MotorState.IDLE

        if blocking:
            _run()
        else:
            self._move_thread = threading.Thread(target=_run, daemon=True)
            self._move_thread.start()

    def _move_with_accel(self, total_steps: int, max_speed_sps: float):
        """Move with trapezoidal acceleration profile."""
        accel_sps2 = self._motion.acceleration_mm_s2 * self._motion.steps_per_mm

        # Minimum speed to start from
        min_speed_sps = max(max_speed_sps * 0.05, 50.0)

        # Calculate steps needed to accelerate to max speed
        # v² = v0² + 2*a*s → s = (v² - v0²) / (2*a)
        accel_steps = int((max_speed_sps**2 - min_speed_sps**2) / (2 * accel_sps2))
        accel_steps = min(accel_steps, total_steps // 2)

        decel_start = total_steps - accel_steps
        current_speed = min_speed_sps

        for step_num in range(total_steps):
            if not self._single_step():
                break

            # Calculate current speed for this step
            if step_num < accel_steps:
                # Accelerating
                current_speed = min(
                    current_speed + accel_sps2 / max(current_speed, 1),
                    max_speed_sps,
                )
            elif step_num >= decel_start:
                # Decelerating
                current_speed = max(
                    current_speed - accel_sps2 / max(current_speed, 1),
                    min_speed_sps,
                )

            time.sleep(1.0 / current_speed)

    def jog(self, toward_home: bool, speed_mm_s: Optional[float] = None):
        """
        Start continuous movement in a direction. Call stop() to halt.

        Args:
            toward_home: Direction of movement.
            speed_mm_s: Speed in mm/s. None = half of max speed.
        """
        if speed_mm_s is None:
            speed_mm_s = self._motion.max_speed_mm_s * 0.5

        speed_sps = speed_mm_s * self._motion.steps_per_mm
        self._stop_event.clear()

        def _run():
            self.enable()
            self._set_direction(toward_home)
            self._state = MotorState.MOVING

            step_delay = 1.0 / speed_sps
            while not self._stop_event.is_set():
                if not self._single_step():
                    # Limit switch hit - stop immediately
                    logger.warning("Jog stopped by limit switch")
                    break
                time.sleep(step_delay)

            self._state = MotorState.IDLE

        self._move_thread = threading.Thread(target=_run, daemon=True)
        self._move_thread.start()

    def home(self, blocking: bool = True):
        """
        Home the stage by moving toward the home switch.

        Moves toward home at homing speed until the home limit switch triggers,
        then backs off slightly and re-approaches slowly for accuracy.
        """
        self._stop_event.clear()

        def _run():
            self.enable()
            self._state = MotorState.HOMING
            home_speed_sps = self._motion.home_speed_mm_s * self._motion.steps_per_mm
            step_delay = 1.0 / home_speed_sps

            logger.info("Homing: moving toward home switch...")

            # Phase 1: Move toward home at homing speed
            self._set_direction(toward_home=True)
            while not self._stop_event.is_set():
                if not self._single_step():
                    # Home switch triggered
                    break
                time.sleep(step_delay)

            if self._stop_event.is_set():
                self._state = MotorState.IDLE
                return

            # Phase 2: Back off 5mm
            backoff_steps = int(5.0 * self._motion.steps_per_mm)
            logger.info("Homing: backing off 5mm...")
            self._set_direction(toward_home=False)
            slow_delay = 1.0 / (home_speed_sps * 0.5)
            for _ in range(backoff_steps):
                if not self._single_step():
                    break
                time.sleep(slow_delay)

            time.sleep(0.2)

            # Phase 3: Slow approach
            logger.info("Homing: slow approach...")
            self._set_direction(toward_home=True)
            crawl_delay = 1.0 / (home_speed_sps * 0.2)
            while not self._stop_event.is_set():
                if not self._single_step():
                    break
                time.sleep(crawl_delay)

            # Set position to zero
            self._position_steps = 0
            self._state = MotorState.IDLE
            logger.info("Homing complete - position zeroed")

        if blocking:
            _run()
        else:
            self._move_thread = threading.Thread(target=_run, daemon=True)
            self._move_thread.start()

    def move_to_mm(self, position_mm: float, blocking: bool = True):
        """Move to an absolute position in millimeters from home."""
        target_steps = int(position_mm * self._motion.steps_per_mm)
        delta = target_steps - self._position_steps
        if delta != 0:
            self.move_steps(delta, blocking=blocking)

    @property
    def state(self) -> MotorState:
        return self._state

    @property
    def position_mm(self) -> float:
        return self._position_steps / self._motion.steps_per_mm

    @property
    def position_steps(self) -> int:
        return self._position_steps

    @property
    def is_moving(self) -> bool:
        return self._state in (MotorState.MOVING, MotorState.HOMING)

    def shutdown(self):
        """Stop motor, disable driver, clean up."""
        self.stop()
        self.disable()
        self._gpio.write(self._pins.step, LOW)
        logger.info("Stepper controller shut down")
