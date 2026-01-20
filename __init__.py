"""
RealSense D455 Visual SLAM Package

A simple Visual SLAM system using Intel RealSense D455 RGB-D camera.
"""

__version__ = "1.0.0"
__author__ = "RealSense SLAM Team"

from .camera_capture import RealSenseD455
from .visual_slam import VisualSLAM

__all__ = ['RealSenseD455', 'VisualSLAM']
