"""
GPIO manager for Raspberry Pi 5 compatibility.

Pi 5 dropped the legacy GPIO interface that RPi.GPIO relies on.
This module provides a unified interface using:
  - gpiod (libgpiod) as the primary backend (works on Pi 4 and 5)
  - lgpio as fallback
  - Mock backend for development/testing off-Pi

All pin numbers use BCM numbering.
"""

import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Constants
HIGH = 1
LOW = 0
PULL_UP = "pull_up"
PULL_DOWN = "pull_down"
PULL_NONE = "none"


class GPIOBackend(ABC):
    """Abstract GPIO backend interface."""

    @abstractmethod
    def setup_output(self, pin: int, initial: int = LOW):
        ...

    @abstractmethod
    def setup_input(self, pin: int, pull: str = PULL_NONE):
        ...

    @abstractmethod
    def write(self, pin: int, value: int):
        ...

    @abstractmethod
    def read(self, pin: int) -> int:
        ...

    @abstractmethod
    def cleanup(self):
        ...


class GpiodBackend(GPIOBackend):
    """Backend using gpiod (libgpiod) - recommended for Pi 5."""

    def __init__(self, chip_path="/dev/gpiochip4"):
        import gpiod
        from gpiod.line_settings import LineSettings, Direction, Bias, Value

        self._gpiod = gpiod
        self._LineSettings = LineSettings
        self._Direction = Direction
        self._Bias = Bias
        self._Value = Value

        # Pi 5 uses gpiochip4 for user GPIO; Pi 4 uses gpiochip0
        for path in [chip_path, "/dev/gpiochip4", "/dev/gpiochip0"]:
            try:
                self._chip = gpiod.Chip(path)
                logger.info(f"gpiod: opened {path}")
                break
            except (FileNotFoundError, PermissionError):
                continue
        else:
            raise RuntimeError("No GPIO chip found. Are you running on a Raspberry Pi?")

        self._output_lines = {}  # pin -> gpiod.Line
        self._input_lines = {}

    def setup_output(self, pin: int, initial: int = LOW):
        val = self._Value.ACTIVE if initial else self._Value.INACTIVE
        settings = self._LineSettings(
            direction=self._Direction.OUTPUT,
            output_value=val,
        )
        request = self._chip.request_lines(
            consumer="geovue_capture",
            config={pin: settings},
        )
        self._output_lines[pin] = request

    def setup_input(self, pin: int, pull: str = PULL_NONE):
        bias_map = {
            PULL_UP: self._Bias.PULL_UP,
            PULL_DOWN: self._Bias.PULL_DOWN,
            PULL_NONE: self._Bias.DISABLED,
        }
        settings = self._LineSettings(
            direction=self._Direction.INPUT,
            bias=bias_map.get(pull, self._Bias.DISABLED),
        )
        request = self._chip.request_lines(
            consumer="geovue_capture",
            config={pin: settings},
        )
        self._input_lines[pin] = request

    def write(self, pin: int, value: int):
        if pin in self._output_lines:
            val = self._Value.ACTIVE if value else self._Value.INACTIVE
            self._output_lines[pin].set_value(pin, val)

    def read(self, pin: int) -> int:
        if pin in self._input_lines:
            val = self._input_lines[pin].get_value(pin)
            return 1 if val == self._Value.ACTIVE else 0
        return 0

    def cleanup(self):
        for req in list(self._output_lines.values()) + list(self._input_lines.values()):
            try:
                req.release()
            except Exception:
                pass
        self._output_lines.clear()
        self._input_lines.clear()
        try:
            self._chip.close()
        except Exception:
            pass


class LgpioBackend(GPIOBackend):
    """Fallback backend using lgpio."""

    def __init__(self):
        import lgpio
        self._lgpio = lgpio
        self._handle = lgpio.gpiochip_open(0)
        logger.info("lgpio: opened gpiochip0")

    def setup_output(self, pin: int, initial: int = LOW):
        self._lgpio.gpio_claim_output(self._handle, pin, initial)

    def setup_input(self, pin: int, pull: str = PULL_NONE):
        flags = 0
        if pull == PULL_UP:
            flags = self._lgpio.SET_PULL_UP
        elif pull == PULL_DOWN:
            flags = self._lgpio.SET_PULL_DOWN
        self._lgpio.gpio_claim_input(self._handle, pin, flags)

    def write(self, pin: int, value: int):
        self._lgpio.gpio_write(self._handle, pin, value)

    def read(self, pin: int) -> int:
        return self._lgpio.gpio_read(self._handle, pin)

    def cleanup(self):
        try:
            self._lgpio.gpiochip_close(self._handle)
        except Exception:
            pass


class MockBackend(GPIOBackend):
    """Mock backend for development and testing off-Pi."""

    def __init__(self):
        self._pins = {}
        logger.warning("GPIO: Using MOCK backend - no real hardware control")

    def setup_output(self, pin: int, initial: int = LOW):
        self._pins[pin] = {"dir": "out", "value": initial}

    def setup_input(self, pin: int, pull: str = PULL_NONE):
        # Default input reads HIGH (pull-up, switch open)
        self._pins[pin] = {"dir": "in", "value": HIGH, "pull": pull}

    def write(self, pin: int, value: int):
        if pin in self._pins:
            self._pins[pin]["value"] = value

    def read(self, pin: int) -> int:
        if pin in self._pins:
            return self._pins[pin]["value"]
        return HIGH  # Default: pull-up, switch open

    def cleanup(self):
        self._pins.clear()


def create_gpio_backend() -> GPIOBackend:
    """Create the best available GPIO backend for this platform."""
    # Try gpiod first (best for Pi 5)
    try:
        return GpiodBackend()
    except Exception as e:
        logger.debug(f"gpiod backend unavailable: {e}")

    # Try lgpio
    try:
        return LgpioBackend()
    except Exception as e:
        logger.debug(f"lgpio backend unavailable: {e}")

    # Fall back to mock
    logger.warning("No real GPIO backend available, using mock")
    return MockBackend()
