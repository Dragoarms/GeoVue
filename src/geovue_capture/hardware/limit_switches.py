"""
Limit switch monitor - runs continuously to protect the stage.

Limit switches are ALWAYS active and override all motion commands.
When a limit switch is triggered:
  1. Motor is stopped IMMEDIATELY
  2. Direction lock is applied (can only move AWAY from the triggered switch)
  3. An event is fired so the GUI can update

Switch wiring: Normally-open (NO) switches with pull-up resistors.
  - Switch OPEN  → pin reads HIGH (1) → safe
  - Switch CLOSED → pin reads LOW (0)  → TRIGGERED / LIMIT HIT
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

from .gpio_manager import GPIOBackend, HIGH, LOW, PULL_UP

logger = logging.getLogger(__name__)


class SwitchState(Enum):
    """Limit switch states."""
    OPEN = auto()      # Not triggered - safe to move
    TRIGGERED = auto()  # Contact made - STOP and restrict direction


class DirectionLock(Enum):
    """Direction restrictions imposed by limit switches."""
    NONE = auto()           # No restriction - both directions OK
    NO_HOME = auto()        # Home switch hit - can only move AWAY from home
    NO_FAR = auto()         # Far switch hit - can only move AWAY (toward home)
    LOCKED = auto()         # Both switches triggered - no movement allowed


class LimitSwitchMonitor:
    """
    Continuously monitors limit switches and enforces safety.

    This runs in a dedicated thread with high priority polling.
    The motor controller MUST check is_move_allowed() before every step.
    """

    # Polling interval in seconds - fast enough to catch switch at any speed
    POLL_INTERVAL_S = 0.001  # 1ms

    def __init__(self, gpio: GPIOBackend, home_pin: int, far_pin: int,
                 ground_pin: Optional[int] = None):
        self._gpio = gpio
        self._home_pin = home_pin
        self._far_pin = far_pin
        self._ground_pin = ground_pin

        # State
        self._home_state = SwitchState.OPEN
        self._far_state = SwitchState.OPEN
        self._direction_lock = DirectionLock.NONE
        self._lock = threading.Lock()

        # Emergency stop flag - set by limit switches, must be explicitly cleared
        self._emergency_stop = threading.Event()

        # Callbacks
        self._on_limit_hit: Optional[Callable[[str, SwitchState], None]] = None
        self._on_limit_cleared: Optional[Callable[[str], None]] = None

        # Monitor thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Setup GPIO
        self._setup_pins()

    def _setup_pins(self):
        """Configure limit switch GPIO pins."""
        # If a GPIO pin is used as ground source for a switch, set it LOW
        if self._ground_pin is not None:
            self._gpio.setup_output(self._ground_pin, LOW)

        # Input pins with pull-up: HIGH = open/safe, LOW = triggered
        self._gpio.setup_input(self._home_pin, PULL_UP)
        self._gpio.setup_input(self._far_pin, PULL_UP)

        logger.info(
            f"Limit switches configured: home=GPIO{self._home_pin}, "
            f"far=GPIO{self._far_pin}"
            + (f", ground=GPIO{self._ground_pin}" if self._ground_pin else "")
        )

    def set_callbacks(self, on_limit_hit=None, on_limit_cleared=None):
        """Set callback functions for limit switch events."""
        self._on_limit_hit = on_limit_hit
        self._on_limit_cleared = on_limit_cleared

    def start(self):
        """Start the limit switch monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="LimitSwitchMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Limit switch monitor started")

    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Limit switch monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop - runs continuously."""
        while self._running:
            self._check_switches()
            time.sleep(self.POLL_INTERVAL_S)

    def _check_switches(self):
        """Read switches and update state."""
        home_raw = self._gpio.read(self._home_pin)
        far_raw = self._gpio.read(self._far_pin)

        home_triggered = (home_raw == LOW)
        far_triggered = (far_raw == LOW)

        with self._lock:
            old_home = self._home_state
            old_far = self._far_state

            # Update home switch state
            if home_triggered:
                self._home_state = SwitchState.TRIGGERED
                if old_home == SwitchState.OPEN:
                    logger.warning("LIMIT: Home switch TRIGGERED - emergency stop")
                    self._emergency_stop.set()
                    if self._on_limit_hit:
                        self._on_limit_hit("home", SwitchState.TRIGGERED)
            else:
                self._home_state = SwitchState.OPEN
                if old_home == SwitchState.TRIGGERED:
                    logger.info("LIMIT: Home switch cleared")
                    if self._on_limit_cleared:
                        self._on_limit_cleared("home")

            # Update far switch state
            if far_triggered:
                self._far_state = SwitchState.TRIGGERED
                if old_far == SwitchState.OPEN:
                    logger.warning("LIMIT: Far switch TRIGGERED - emergency stop")
                    self._emergency_stop.set()
                    if self._on_limit_hit:
                        self._on_limit_hit("far", SwitchState.TRIGGERED)
            else:
                self._far_state = SwitchState.OPEN
                if old_far == SwitchState.TRIGGERED:
                    logger.info("LIMIT: Far switch cleared")
                    if self._on_limit_cleared:
                        self._on_limit_cleared("far")

            # Update direction lock
            if home_triggered and far_triggered:
                self._direction_lock = DirectionLock.LOCKED
            elif home_triggered:
                self._direction_lock = DirectionLock.NO_HOME
            elif far_triggered:
                self._direction_lock = DirectionLock.NO_FAR
            else:
                self._direction_lock = DirectionLock.NONE
                # Clear emergency stop only when both switches are clear
                self._emergency_stop.clear()

    def is_move_allowed(self, direction_is_home: bool) -> bool:
        """
        Check if movement in the given direction is allowed.

        This MUST be called before every step pulse.

        Args:
            direction_is_home: True if moving toward home, False if moving away.

        Returns:
            True if movement is safe, False if blocked by a limit switch.
        """
        with self._lock:
            if self._direction_lock == DirectionLock.LOCKED:
                return False
            if self._direction_lock == DirectionLock.NO_HOME and direction_is_home:
                return False
            if self._direction_lock == DirectionLock.NO_FAR and not direction_is_home:
                return False
            return True

    @property
    def emergency_stopped(self) -> bool:
        """True if a limit switch has triggered and not yet cleared."""
        return self._emergency_stop.is_set()

    @property
    def home_triggered(self) -> bool:
        with self._lock:
            return self._home_state == SwitchState.TRIGGERED

    @property
    def far_triggered(self) -> bool:
        with self._lock:
            return self._far_state == SwitchState.TRIGGERED

    @property
    def direction_lock(self) -> DirectionLock:
        with self._lock:
            return self._direction_lock

    def read_raw(self) -> dict:
        """Read raw switch states for diagnostics."""
        return {
            "home_pin": self._gpio.read(self._home_pin),
            "far_pin": self._gpio.read(self._far_pin),
            "home_state": self._home_state.name,
            "far_state": self._far_state.name,
            "direction_lock": self._direction_lock.name,
            "emergency_stop": self._emergency_stop.is_set(),
        }
