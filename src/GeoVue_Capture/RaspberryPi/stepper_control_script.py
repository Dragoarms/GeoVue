#!/usr/bin/env python3
"""
Stepper Motor Control with Limit Switch Safety
Controls stepper motor with arrow keys and monitors limit switches

Wiring:
- Pin 9 (GND) - All negative signals
- Pin 11 (GPIO 17) - PUL+ (Step pulses)
- Pin 13 (GPIO 27) - DIR+ (Direction control) 
- Pin 15 (GPIO 22) - ENA+ (Enable/Disable)

Limit Switches:
- Switch 1: Pin 33 (GPIO 13) ←→ Pin 35 (GPIO 19) 
- Switch 2: Pin 37 (GPIO 26) ←→ Pin 39 (GND)

Controls:
- Left Arrow: Move left (DIR LOW)
- Right Arrow: Move right (DIR HIGH)  
- Up Arrow: Increase speed
- Down Arrow: Decrease speed
- Space: Stop motor
- ESC: Exit program
"""

import RPi.GPIO as GPIO
import time
import threading
import sys
import select
import termios
import tty

# Stepper motor pins
PUL_PIN = 17    # GPIO 17 (Pin 11) - Step pulses
DIR_PIN = 27    # GPIO 27 (Pin 13) - Direction 
ENA_PIN = 22    # GPIO 22 (Pin 15) - Enable

# Limit switch pins  
SWITCH1_OUTPUT_PIN = 13  # GPIO 13 (Pin 33) - acts as ground for switch 1
SWITCH1_INPUT_PIN = 19   # GPIO 19 (Pin 35) - reads switch 1
SWITCH2_INPUT_PIN = 26   # GPIO 26 (Pin 37) - reads switch 2

# Motor control variables
motor_running = False
motor_direction = True  # True = right, False = left
motor_speed = 1000      # Steps per second (start value)
min_speed = 100         # Minimum speed
max_speed = 5000        # Maximum speed
speed_increment = 200   # Speed change per keypress

# Safety flags
limit_switch_triggered = False
emergency_stop = False

class KeyboardReader:
    """Non-blocking keyboard input reader"""
    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Get a single keypress without blocking"""
        if select.select([sys.stdin], [], [], 0.01) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
            # Handle arrow keys (they send escape sequences)
            if key == '\x1b':  # ESC sequence
                key += sys.stdin.read(2)
            return key
        return None

def setup_gpio():
    """Initialize all GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Stepper motor pins
    GPIO.setup(PUL_PIN, GPIO.OUT)
    GPIO.setup(DIR_PIN, GPIO.OUT) 
    GPIO.setup(ENA_PIN, GPIO.OUT)
    
    # Initialize stepper pins
    GPIO.output(PUL_PIN, GPIO.LOW)
    GPIO.output(DIR_PIN, GPIO.HIGH)  # Default direction
    GPIO.output(ENA_PIN, GPIO.LOW)   # Enable motor (LOW = enabled for most drivers)
    
    # Limit switch pins
    GPIO.setup(SWITCH1_OUTPUT_PIN, GPIO.OUT)
    GPIO.output(SWITCH1_OUTPUT_PIN, GPIO.LOW)  # Acts as ground
    GPIO.setup(SWITCH1_INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SWITCH2_INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    print("GPIO setup complete:")
    print(f"Stepper: PUL={PUL_PIN}, DIR={DIR_PIN}, ENA={ENA_PIN}")
    print(f"Limits: SW1={SWITCH1_INPUT_PIN}, SW2={SWITCH2_INPUT_PIN}")

def check_limit_switches():
    """Check if any limit switch is triggered"""
    global limit_switch_triggered, motor_running
    
    switch1_state = GPIO.input(SWITCH1_INPUT_PIN)
    switch2_state = GPIO.input(SWITCH2_INPUT_PIN)
    
    if switch1_state == GPIO.LOW:  # Switch 1 triggered
        if not limit_switch_triggered:
            print(f"\n*** LIMIT SWITCH 1 TRIGGERED - EMERGENCY STOP ***")
            limit_switch_triggered = True
            motor_running = False
        return True
        
    if switch2_state == GPIO.LOW:  # Switch 2 triggered  
        if not limit_switch_triggered:
            print(f"\n*** LIMIT SWITCH 2 TRIGGERED - EMERGENCY STOP ***")
            limit_switch_triggered = True
            motor_running = False
        return True
    
    # Reset flag if switches are released
    if limit_switch_triggered and switch1_state == GPIO.HIGH and switch2_state == GPIO.HIGH:
        print("Limit switches cleared - motor can restart")
        limit_switch_triggered = False
    
    return False

def stepper_thread():
    """Thread function to generate stepper pulses"""
    global motor_running, motor_direction, motor_speed, emergency_stop
    
    while not emergency_stop:
        if motor_running and not limit_switch_triggered:
            # Set direction
            GPIO.output(DIR_PIN, GPIO.HIGH if motor_direction else GPIO.LOW)
            
            # Calculate delay for current speed (steps per second)
            step_delay = 1.0 / (motor_speed * 2)  # *2 because we need HIGH and LOW
            
            # Generate one step pulse
            GPIO.output(PUL_PIN, GPIO.HIGH)
            time.sleep(step_delay)
            GPIO.output(PUL_PIN, GPIO.LOW) 
            time.sleep(step_delay)
            
            # Check limit switches during movement
            check_limit_switches()
        else:
            time.sleep(0.01)  # Small delay when not running

def display_status():
    """Display current motor status"""
    direction_text = "RIGHT" if motor_direction else "LEFT"
    status_text = "RUNNING" if motor_running else "STOPPED"
    
    print(f"\rStatus: {status_text} | Direction: {direction_text} | Speed: {motor_speed} steps/sec", end="", flush=True)

def main():
    """Main control loop"""
    global motor_running, motor_direction, motor_speed, emergency_stop
    
    print("Stepper Motor Control with Limit Switches")
    print("=" * 50)
    print("Controls:")
    print("  Left Arrow  : Move left")
    print("  Right Arrow : Move right") 
    print("  Up Arrow    : Increase speed")
    print("  Down Arrow  : Decrease speed")
    print("  Space       : Stop motor")
    print("  ESC         : Exit program")
    print("=" * 50)
    
    setup_gpio()
    
    # Start stepper control thread
    stepper_control = threading.Thread(target=stepper_thread, daemon=True)
    stepper_control.start()
    
    # Initialize keyboard reader
    try:
        keyboard = KeyboardReader()
        
        print("\nMotor ready. Use arrow keys to control...")
        display_status()
        
        while True:
            key = keyboard.get_key()
            
            if key:
                # ESC key to exit
                if key == '\x1b[' or key == '\x1b':
                    break
                
                # Arrow keys  
                elif key == '\x1b[D':  # Left arrow
                    if not limit_switch_triggered:
                        motor_direction = False
                        motor_running = True
                        print(f"\nMoving LEFT at {motor_speed} steps/sec")
                
                elif key == '\x1b[C':  # Right arrow
                    if not limit_switch_triggered:
                        motor_direction = True  
                        motor_running = True
                        print(f"\nMoving RIGHT at {motor_speed} steps/sec")
                
                elif key == '\x1b[A':  # Up arrow - increase speed
                    motor_speed = min(motor_speed + speed_increment, max_speed)
                    print(f"\nSpeed increased to {motor_speed} steps/sec")
                
                elif key == '\x1b[B':  # Down arrow - decrease speed  
                    motor_speed = max(motor_speed - speed_increment, min_speed)
                    print(f"\nSpeed decreased to {motor_speed} steps/sec")
                
                elif key == ' ':  # Space - stop
                    motor_running = False
                    print("\nMotor STOPPED")
                
                elif key == 'q' or key == 'Q':  # Alternative quit
                    break
                    
                display_status()
            
            # Continuous limit switch monitoring
            check_limit_switches()
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\n\nShutting down...")
        emergency_stop = True
        motor_running = False
        
        # Disable motor
        GPIO.output(ENA_PIN, GPIO.HIGH)  # Disable motor
        GPIO.output(PUL_PIN, GPIO.LOW)
        
        # Clean up
        GPIO.cleanup()
        print("GPIO cleaned up. Goodbye!")

if __name__ == "__main__":
    main()