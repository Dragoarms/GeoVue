#!/usr/bin/env python3
"""
GeoVue Camera Control Application
Integrated camera control, stage management, and image processing for chip tray photography
"""

import os
import sys
import json
import time
import socket
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import queue
import logging
import io
import re
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and Constants
# ============================================================================

class Config:
    """Application configuration"""
    # Paths
    BASE_DIR = Path("/home/pi/GeoVue")
    STORAGE_DIR = BASE_DIR / "storage"
    IMAGES_DIR = STORAGE_DIR / "Images to Process"
    DELETED_DIR = STORAGE_DIR / "Deleted files"
    TEMP_DIR = STORAGE_DIR / "temp"
    
    # Stage control
    STAGE_SERVER_HOST = "127.0.0.1"
    STAGE_SERVER_PORT = 5055
    HOME_POSITION = 0
    CALIBRATION_POSITION_MM = 150  # 15cm in mm
    STEPS_PER_MM = 80  # Calibrate this for your motor
    
    # Camera settings
    PANORAMA_MODE_COORDS = (0.12, 0.92)  # Normalized coordinates for panorama button
    ZOOM_1X_COORDS = (0.50, 0.50)  # Center tap for 1x zoom
    USB_HUB_PATH = "1-1"  # USB hub path for uhubctl
    
    # Image processing
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    COMPARTMENT_COUNT = 20
    DEPTH_INCREMENT = 20  # meters
    
    # UI Settings
    WINDOW_WIDTH = 1024
    WINDOW_HEIGHT = 768
    PREVIEW_WIDTH = 400
    PREVIEW_HEIGHT = 600

# ============================================================================
# Stage Control Client
# ============================================================================

class StageController:
    """Client for controlling the stepper motor stage via TCP"""
    
    def __init__(self, host=Config.STAGE_SERVER_HOST, port=Config.STAGE_SERVER_PORT):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__ + '.StageController')
        
    def send_command(self, command):
        """Send command to stage server"""
        try:
            with socket.create_connection((self.host, self.port), timeout=5) as sock:
                sock.sendall(command.encode())
                response = sock.recv(16).decode().strip()
                return response == "OK"
        except Exception as e:
            self.logger.error(f"Stage command failed: {e}")
            return False
    
    def move_left(self):
        """Move stage left"""
        return self.send_command("LEFT")
    
    def move_right(self):
        """Move stage right"""
    
    def stop(self):
        """Stop stage movement"""
        return self.send_command("STOP")
    
    def go_home(self):
        """Return stage to home position (left limit switch)"""
        return self.send_command("HOME")
    
    def step(self, steps):
        """Move stage by specific number of steps"""
        return self.send_command(f"STEP:{steps}")
    
    def move_to_calibration(self):
        """Move to calibration/focus position"""
        steps = int(Config.CALIBRATION_POSITION_MM * Config.STEPS_PER_MM)
        return self.step(steps)

# ============================================================================
# Camera Controller
# ============================================================================

class CameraController:
    """Enhanced phone camera controller with USB power and panorama support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.CameraController')
        self.device_connected = self.check_device()
        
    def check_device(self):
        """Check if ADB device is connected"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            devices = [line for line in lines if line.strip() and 'device' in line]
            return len(devices) > 0
        except:
            return False
    
    def usb_power(self, on=True):
        """Toggle USB power via uhubctl"""
        try:
            action = "1" if on else "0"
            cmd = ["uhubctl", "-l", Config.USB_HUB_PATH, "-a", action]
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"USB power {'on' if on else 'off'}")
            time.sleep(0.5 if on else 0.1)
            return True
        except Exception as e:
            self.logger.error(f"USB power control failed: {e}")
            return False
    
    def run_adb_command(self, command):
        """Run an ADB shell command"""
        try:
            full_command = f"adb shell {command}"
            result = subprocess.run(full_command.split(), check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ADB command failed: {e}")
            return False
    
    def open_camera(self):
        """Open camera app and switch to panorama mode"""
        self.usb_power(True)
        time.sleep(1)
        
        # Launch camera
        self.run_adb_command("am start -a android.media.action.IMAGE_CAPTURE")
        time.sleep(2)
        
        # Try to tap panorama mode button
        try:
            # Get screen dimensions
            result = subprocess.run(['adb', 'shell', 'wm', 'size'], 
                                  capture_output=True, text=True)
            dimensions = result.stdout.split(':')[1].strip()
            width, height = map(int, dimensions.split('x'))
            
            # Tap panorama button
            pano_x = int(Config.PANORAMA_MODE_COORDS[0] * width)
            pano_y = int(Config.PANORAMA_MODE_COORDS[1] * height)
            self.run_adb_command(f"input tap {pano_x} {pano_y}")
            
            # Ensure 1x zoom
            zoom_x = int(Config.ZOOM_1X_COORDS[0] * width)
            zoom_y = int(Config.ZOOM_1X_COORDS[1] * height)
            self.run_adb_command(f"input tap {zoom_x} {zoom_y}")
            
            self.logger.info("Camera opened in panorama mode")
        except Exception as e:
            self.logger.error(f"Failed to set panorama mode: {e}")
    
    def focus_center(self):
        """Tap center to focus"""
        try:
            result = subprocess.run(['adb', 'shell', 'wm', 'size'], 
                                  capture_output=True, text=True)
            dimensions = result.stdout.split(':')[1].strip()
            width, height = map(int, dimensions.split('x'))
            
            center_x = width // 2
            center_y = height // 2
            
            self.run_adb_command(f"input tap {center_x} {center_y}")
            self.logger.info("Focused on center")
            return True
        except Exception as e:
            self.logger.error(f"Focus failed: {e}")
            return False
    
    def lock_focus(self):
        """Long press to lock focus"""
        try:
            result = subprocess.run(['adb', 'shell', 'wm', 'size'], 
                                  capture_output=True, text=True)
            dimensions = result.stdout.split(':')[1].strip()
            width, height = map(int, dimensions.split('x'))
            
            center_x = width // 2
            center_y = height // 2
            
            # Long press for 1.5 seconds
            self.run_adb_command(f"input swipe {center_x} {center_y} {center_x} {center_y} 1500")
            self.logger.info("Focus locked")
            return True
        except Exception as e:
            self.logger.error(f"Lock focus failed: {e}")
            return False
    
    def take_photo(self):
        """Trigger camera shutter"""
        return self.run_adb_command("input keyevent KEYCODE_CAMERA")
    
    def get_screenshot(self):
        """Get current phone screen as PIL Image"""
        try:
            result = subprocess.run(['adb', 'exec-out', 'screencap', '-p'], 
                                  capture_output=True)
            if result.returncode == 0:
                return Image.open(io.BytesIO(result.stdout))
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
        return None
    
    def pull_latest_photo(self, destination):
        """Pull the most recent photo from phone"""
        try:
            # Find latest file in DCIM
            result = subprocess.run(
                ['adb', 'shell', 'ls', '-t', '/sdcard/DCIM/Camera/', '|', 'head', '-1'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                latest_file = result.stdout.strip()
                if latest_file:
                    source = f"/sdcard/DCIM/Camera/{latest_file}"
                    subprocess.run(['adb', 'pull', source, destination], check=True)
                    self.logger.info(f"Pulled photo: {latest_file}")
                    return destination
        except Exception as e:
            self.logger.error(f"Failed to pull photo: {e}")
        return None

# ============================================================================
# ArUco Detection Module
# ============================================================================

class ArucoDetector:
    """Simple ArUco marker detector for live preview"""
    
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(Config.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def detect_and_draw(self, image):
        """Detect ArUco markers and draw them on image"""
        if isinstance(image, Image.Image):
            # Convert PIL to OpenCV
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(image)
        
        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # Convert back to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb), len(ids) if ids is not None else 0

# ============================================================================
# Data Manager
# ============================================================================

class DataManager:
    """Manages session data and file operations"""
    
    def __init__(self):
        self.session_file = Config.STORAGE_DIR / "session_state.json"
        self.session_data = self.load_session()
        self.processed_today = []
        
    def load_session(self):
        """Load session state"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "hole_id": "",
            "depth_from": 0,
            "depth_to": 20,
            "compartment_interval": 1
        }
    
    def save_session(self):
        """Save session state"""
        try:
            Config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def save_image(self, image_path, hole_id, depth_from, depth_to, moisture):
        """Save and rename image with metadata"""
        try:
            # Create directory structure
            project_code = hole_id[:2] if len(hole_id) >= 2 else "XX"
            save_dir = Config.IMAGES_DIR / project_code / hole_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{hole_id}_{depth_from}-{depth_to}m_{moisture}_{timestamp}.jpg"
            dest_path = save_dir / filename
            
            # Move or copy file
            if Path(image_path).exists():
                shutil.move(str(image_path), str(dest_path))
                
                # Add to processed list
                self.processed_today.append({
                    "hole_id": hole_id,
                    "depth": f"{depth_from}-{depth_to}m",
                    "moisture": moisture,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "path": str(dest_path)
                })
                
                logger.info(f"Saved image: {dest_path}")
                return str(dest_path)
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
        return None
    
    def delete_image(self, image_path):
        """Move image to deleted folder"""
        try:
            deleted_dir = Config.DELETED_DIR / datetime.now().strftime("%Y%m%d")
            deleted_dir.mkdir(parents=True, exist_ok=True)
            
            dest = deleted_dir / Path(image_path).name
            shutil.move(image_path, str(dest))
            
            # Remove from processed list
            self.processed_today = [
                p for p in self.processed_today if p.get('path') != image_path
            ]
            
            logger.info(f"Moved to deleted: {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            return False

# ============================================================================
# Main Application GUI
# ============================================================================

class GeoVueCameraApp:
    """Main application window"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GeoVue Camera Control")
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        
        # Initialize components
        self.stage = StageController()
        self.camera = CameraController()
        self.aruco = ArucoDetector()
        self.data_manager = DataManager()
        
        # Preview update thread control
        self.preview_running = False
        self.preview_thread = None
        
        # Setup UI
        self.setup_ui()
        
        # Load last session
        self.load_last_session()
        
        # Start preview
        self.start_preview()
    
    def setup_ui(self):
        """Create the user interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ========== Left Panel: Controls ==========
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Data entry fields
        ttk.Label(control_frame, text="Hole ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.hole_id_var = tk.StringVar()
        self.hole_id_entry = ttk.Entry(control_frame, textvariable=self.hole_id_var, width=15)
        self.hole_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(control_frame, text="Depth Range:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.depth_var = tk.StringVar()
        depth_values = [f"{i}-{i+20}" for i in range(0, 500, 20)]
        self.depth_combo = ttk.Combobox(control_frame, textvariable=self.depth_var, 
                                        values=depth_values, width=13, state='readonly')
        self.depth_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(control_frame, text="Moisture:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.moisture_var = tk.StringVar(value="Dry")
        moisture_frame = ttk.Frame(control_frame)
        moisture_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        for moisture in ["Wet", "Dry", "Mixed"]:
            ttk.Radiobutton(moisture_frame, text=moisture, variable=self.moisture_var, 
                           value=moisture).pack(side=tk.LEFT)
        
        # Stage controls
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, 
                                                               columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Stage Control:", font=('', 10, 'bold')).grid(row=4, column=0, 
                                                                                     columnspan=2, pady=5)
        
        stage_frame = ttk.Frame(control_frame)
        stage_frame.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.left_btn = ttk.Button(stage_frame, text="◀ Left", command=self.move_left, width=10)
        self.left_btn.pack(side=tk.LEFT, padx=2)
        
        self.home_btn = ttk.Button(stage_frame, text="⌂ Home", command=self.go_home, width=10)
        self.home_btn.pack(side=tk.LEFT, padx=2)
        
        self.right_btn = ttk.Button(stage_frame, text="Right ▶", command=self.move_right, width=10)
        self.right_btn.pack(side=tk.LEFT, padx=2)
        
        # Camera controls
        ttk.Separator(control_frame, orient='horizontal').grid(row=6, column=0, 
                                                               columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.take_photo_btn = ttk.Button(control_frame, text="📷 Take Picture", 
                                         command=self.take_photo_sequence, 
                                         style='Accent.TButton')
        self.take_photo_btn.grid(row=7, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, 
                 foreground="green").grid(row=8, column=0, columnspan=2, pady=5)
        
        # ========== Center Panel: Preview ==========
        preview_frame = ttk.LabelFrame(main_frame, text="Live Preview", padding="5")
        preview_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(expand=True, fill=tk.BOTH)
        
        self.marker_count_var = tk.StringVar(value="Markers: 0")
        ttk.Label(preview_frame, textvariable=self.marker_count_var).pack()
        
        # ========== Right Panel: Processed Today ==========
        processed_frame = ttk.LabelFrame(main_frame, text="Processed Today", padding="5")
        processed_frame.grid(row=0, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Treeview for processed images
        columns = ('Time', 'Hole ID', 'Depth', 'Moisture')
        self.processed_tree = ttk.Treeview(processed_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.processed_tree.heading(col, text=col)
            self.processed_tree.column(col, width=80)
        
        self.processed_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(processed_frame, orient=tk.VERTICAL, command=self.processed_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.processed_tree.configure(yscrollcommand=scrollbar.set)
        
        # Context menu for deletion
        self.processed_tree.bind("<Button-3>", self.show_context_menu)
        
        # ========== Bottom Panel: Gallery ==========
        gallery_frame = ttk.LabelFrame(main_frame, text="Recent Images", padding="5")
        gallery_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Canvas for thumbnail gallery
        self.gallery_canvas = tk.Canvas(gallery_frame, height=100)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        gallery_scrollbar = ttk.Scrollbar(gallery_frame, orient=tk.HORIZONTAL, 
                                         command=self.gallery_canvas.xview)
        gallery_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.gallery_canvas.configure(xscrollcommand=gallery_scrollbar.set)
        
        self.gallery_frame_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_frame_inner, anchor='nw')
        
    def load_last_session(self):
        """Load last session data"""
        session = self.data_manager.session_data
        self.hole_id_var.set(session.get("hole_id", ""))
        
        depth_from = session.get("depth_from", 0)
        depth_to = session.get("depth_to", 20)
        self.depth_var.set(f"{depth_from}-{depth_to}")
    
    def save_session(self):
        """Save current session data"""
        depth_parts = self.depth_var.get().split('-')
        self.data_manager.session_data = {
            "hole_id": self.hole_id_var.get(),
            "depth_from": int(depth_parts[0]) if depth_parts else 0,
            "depth_to": int(depth_parts[1]) if len(depth_parts) > 1 else 20,
            "compartment_interval": 1
        }
        self.data_manager.save_session()
    
    def start_preview(self):
        """Start the camera preview thread"""
        if not self.preview_running:
            self.preview_running = True
            self.preview_thread = threading.Thread(target=self.update_preview, daemon=True)
            self.preview_thread.start()
    
    def update_preview(self):
        """Update preview with screenshot and ArUco detection"""
        while self.preview_running:
            try:
                # Get screenshot
                img = self.camera.get_screenshot()
                if img:
                    # Resize for preview
                    img.thumbnail((Config.PREVIEW_WIDTH, Config.PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
                    
                    # Detect ArUco markers
                    img_with_markers, marker_count = self.aruco.detect_and_draw(img)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img_with_markers)
                    
                    # Update UI in main thread
                    self.root.after(0, self.update_preview_ui, photo, marker_count)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Preview update error: {e}")
                time.sleep(2)
    
    def update_preview_ui(self, photo, marker_count):
        """Update preview UI elements (must be called from main thread)"""
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # Keep reference
        self.marker_count_var.set(f"Markers: {marker_count}")
    
    def move_left(self):
        """Move stage left"""
        self.status_var.set("Moving left...")
        if self.stage.move_left():
            self.status_var.set("Stage moving left")
        else:
            self.status_var.set("Stage command failed")
    
    def move_right(self):
        """Move stage right"""
        self.status_var.set("Moving right...")
        if self.stage.move_right():
            self.status_var.set("Stage moving right")
        else:
            self.status_var.set("Stage command failed")
    
    def go_home(self):
        """Return stage to home position"""
        self.status_var.set("Returning home...")
        if self.stage.go_home():
            self.status_var.set("Stage at home position")
        else:
            self.status_var.set("Home command failed")
    
    def take_photo_sequence(self):
        """Execute the photo capture sequence"""
        # Validate inputs
        if not self.hole_id_var.get():
            messagebox.showwarning("Input Error", "Please enter Hole ID")
            return
        
        if not self.depth_var.get():
            messagebox.showwarning("Input Error", "Please select depth range")
            return
        
        # Disable button during capture
        self.take_photo_btn.configure(state='disabled')
        
        # Run sequence in thread
        thread = threading.Thread(target=self._photo_sequence_thread)
        thread.start()
    
    def _photo_sequence_thread(self):
        """Photo capture sequence (runs in thread)"""
        try:
            # 1. Turn on USB power and go home
            self.update_status("Powering on camera...")
            self.camera.usb_power(True)
            
            self.update_status("Moving to home position...")
            self.stage.go_home()
            time.sleep(2)
            
            # 2. Move to calibration position
            self.update_status("Moving to calibration position...")
            self.stage.move_to_calibration()
            time.sleep(2)
            
            # 3. Open camera in panorama mode
            self.update_status("Opening camera...")
            self.camera.open_camera()
            time.sleep(3)
            
            # 4. Focus and lock
            self.update_status("Focusing...")
            self.camera.focus_center()
            time.sleep(1)
            self.camera.lock_focus()
            time.sleep(1)
            
            # 5. Take photo
            self.update_status("Taking photo...")
            self.camera.take_photo()
            time.sleep(3)
            
            # 6. Pull and save photo
            self.update_status("Saving photo...")
            temp_path = Config.TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            
            if self.camera.pull_latest_photo(str(temp_path)):
                # Get metadata
                hole_id = self.hole_id_var.get()
                depth_parts = self.depth_var.get().split('-')
                depth_from = int(depth_parts[0])
                depth_to = int(depth_parts[1])
                moisture = self.moisture_var.get()
                
                # Save with metadata
                saved_path = self.data_manager.save_image(
                    str(temp_path), hole_id, depth_from, depth_to, moisture
                )
                
                if saved_path:
                    # Update processed list
                    self.root.after(0, self.update_processed_list)
                    self.root.after(0, self.update_gallery)
                    
                    # Auto-increment depth
                    new_depth_from = depth_to
                    new_depth_to = depth_to + Config.DEPTH_INCREMENT
                    self.root.after(0, lambda: self.depth_var.set(f"{new_depth_from}-{new_depth_to}"))
                    
                    # Save session
                    self.save_session()
                    
                    self.update_status("Photo saved successfully!")
                else:
                    self.update_status("Failed to save photo")
            else:
                self.update_status("Failed to retrieve photo")
            
        except Exception as e:
            logger.error(f"Photo sequence error: {e}")
            self.update_status(f"Error: {str(e)}")
        
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.take_photo_btn.configure(state='normal'))
    
    def update_status(self, message):
        """Update status in main thread"""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def update_processed_list(self):
        """Update the processed today list"""
        # Clear existing items
        for item in self.processed_tree.get_children():
            self.processed_tree.delete(item)
        
        # Add new items
        for entry in self.data_manager.processed_today:
            self.processed_tree.insert('', 'end', values=(
                entry['time'],
                entry['hole_id'],
                entry['depth'],
                entry['moisture']
            ), tags=(entry['path'],))
    
    def update_gallery(self):
        """Update the thumbnail gallery"""
        # Clear existing thumbnails
        for widget in self.gallery_frame_inner.winfo_children():
            widget.destroy()
        
        # Add thumbnails for recent images
        for i, entry in enumerate(self.data_manager.processed_today[-10:]):  # Last 10 images
            try:
                img = Image.open(entry['path'])
                img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                label = ttk.Label(self.gallery_frame_inner, image=photo)
                label.image = photo  # Keep reference
                label.grid(row=0, column=i, padx=2)
                
                # Bind click event
                label.bind("<Button-1>", lambda e, p=entry['path']: self.show_full_image(p))
                
            except Exception as e:
                logger.error(f"Failed to load thumbnail: {e}")
        
        # Update scroll region
        self.gallery_frame_inner.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
    
    def show_context_menu(self, event):
        """Show context menu for tree item"""
        item = self.processed_tree.identify('item', event.x, event.y)
        if item:
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="Delete", 
                           command=lambda: self.delete_image(item))
            menu.post(event.x_root, event.y_root)
    
    def delete_image(self, item):
        """Delete selected image"""
        tags = self.processed_tree.item(item, 'tags')
        if tags:
            image_path = tags[0]
            if messagebox.askyesno("Delete Image", "Move this image to deleted folder?"):
                if self.data_manager.delete_image(image_path):
                    self.update_processed_list()
                    self.update_gallery()
    
    def show_full_image(self, image_path):
        """Show full size image in new window"""
        try:
            window = tk.Toplevel(self.root)
            window.title(os.path.basename(image_path))
            
            img = Image.open(image_path)
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = ttk.Label(window, image=photo)
            label.image = photo
            label.pack()
            
        except Exception as e:
            logger.error(f"Failed to show image: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.preview_running = False
        self.save_session()
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Ensure running as root for GPIO access
    if os.geteuid() != 0:
        print("This application requires root privileges for GPIO and USB control.")
        print("Please run with sudo.")
        sys.exit(1)
    
    # Create base directories
    Config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    Config.DELETED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start application
    app = GeoVueCameraApp()
    app.run()
