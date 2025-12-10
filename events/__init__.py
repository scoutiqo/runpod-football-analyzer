# events package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from events import infer_possession, detect_passes

__all__ = ['infer_possession', 'detect_passes']
