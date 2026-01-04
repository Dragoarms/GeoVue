# tests/__init__.py
"""Test suite for QAQC Manager."""

import os
import sys
from pathlib import Path

# Fix import paths
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test constants that match your application
TEST_HOLE_IDS = ["AB1234", "CD5678", "EF9012"]
TEST_DEPTHS = [0, 20, 18, 115, 120]
TEST_COMPARTMENT_INTERVAL = [1, 2]

# Create test data directory if needed
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

print(f"Test suite initialized. Project root: {PROJECT_ROOT}")