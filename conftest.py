import os
import sys


repo_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
