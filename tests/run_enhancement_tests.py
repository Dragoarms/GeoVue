#!/usr/bin/env python3
"""
Test runner for JSON register manager enhancement tests.

This script provides a way to run the enhancement tests with proper environment setup
and dependency checking.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import openpyxl
    except ImportError:
        missing_deps.append('openpyxl')
    
    return missing_deps

def install_dependencies(deps):
    """Attempt to install missing dependencies."""
    print(f"Attempting to install missing dependencies: {', '.join(deps)}")
    
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
            return False
    
    return True

def run_tests(test_file=None, verbose=True):
    """Run the enhancement tests."""
    test_dir = Path(__file__).parent
    
    if test_file is None:
        test_file = test_dir / "test_json_register_manager_enhancements.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    print(f"Running tests from: {test_file}")
    
    # Try to run with unittest
    try:
        if verbose:
            result = subprocess.run([
                sys.executable, '-m', 'unittest', '-v', str(test_file)
            ], cwd=test_dir.parent, capture_output=False)
        else:
            result = subprocess.run([
                sys.executable, str(test_file)
            ], cwd=test_dir.parent, capture_output=False)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_validation():
    """Run the test structure validation."""
    test_dir = Path(__file__).parent
    validation_script = test_dir / "validate_test_structure.py"
    
    if not validation_script.exists():
        print("Validation script not found")
        return False
    
    print("Running test structure validation...")
    try:
        result = subprocess.run([
            sys.executable, str(validation_script)
        ], cwd=test_dir.parent, capture_output=False)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running validation: {e}")
        return False

def main():
    """Main function."""
    print("JSON Register Manager Enhancement Tests Runner")
    print("=" * 50)
    
    # First run validation
    print("\n1. Running test structure validation...")
    validation_success = run_validation()
    
    if not validation_success:
        print("Test validation failed. Please check test structure.")
        return False
    
    print("\n2. Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        
        # Ask user if they want to install
        response = input("Do you want to attempt to install missing dependencies? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies(missing_deps):
                print("Failed to install some dependencies. Tests may fail.")
        else:
            print("Dependencies not installed. Tests may fail.")
    
    print("\n3. Running enhancement tests...")
    test_success = run_tests()
    
    if test_success:
        print("\nAll tests completed successfully!")
        return True
    else:
        print("\nSome tests failed or encountered errors.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)