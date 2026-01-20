"""
RealSense D455 Camera Capture Module
Captures RGB-D data from Intel RealSense D455 camera
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time


class RealSenseD455:
    """RealSense D455 camera wrapper for RGB-D data capture"""
    
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense D455 camera
        
        Args:
            width: Image width (default: 640)
            height: Image height (default: 480)
            fps: Frames per second (default: 30)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = 0.001
        
    def initialize(self):
        """Initialize the camera pipeline"""
        try:
            # Create a pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Enable IMU streams
            self.config.enable_stream(rs.stream.accel)
            self.config.enable_stream(rs.stream.gyro)
            
            # Start the pipeline
            profile = self.pipeline.start(self.config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale: {self.depth_scale}")
            
            # Create align object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            print("RealSense D455 initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_frame(self):
        """
        Get aligned RGB and depth frames and IMU data
        
        Returns:
            tuple: (color_image, depth_image, depth_colormap, accel, gyro, timestamp)
        """
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            # Get IMU frames
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            
            if not color_frame or not depth_frame:
                return None, None, None, None, None, 0
            
            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply depth scale to convert to meters
            depth_image = depth_image * self.depth_scale
            
            # Create depth colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0/5.0),
                cv2.COLORMAP_JET
            )
            
            # Get IMU data
            accel_data = None
            if accel_frame:
                accel = accel_frame.as_motion_frame().get_motion_data()
                accel_data = np.array([accel.x, accel.y, accel.z])
                
            gyro_data = None
            if gyro_frame:
                gyro = gyro_frame.as_motion_frame().get_motion_data()
                gyro_data = np.array([gyro.x, gyro.y, gyro.z])
            
            timestamp = frames.get_timestamp() / 1000.0 if frames else time.time()
            
            return color_image, depth_image, depth_colormap, accel_data, gyro_data, timestamp
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None, None, None, None, 0
    
    def get_camera_intrinsics(self):
        """
        Get camera intrinsics for SLAM
        
        Returns:
            dict: Camera intrinsics parameters
        """
        try:
            # Get the color stream profile
            profile = self.pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            intrinsics = color_profile.get_intrinsics()
            
            return {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'width': intrinsics.width,
                'height': intrinsics.height,
                'depth_scale': self.depth_scale
            }
        except Exception as e:
            print(f"Error getting intrinsics: {e}")
            return None
    
    def stop(self):
        """Stop the camera pipeline"""
        if self.pipeline:
            self.pipeline.stop()
            print("Camera stopped")
    
    def __del__(self):
        """Destructor to ensure camera is stopped"""
        self.stop()


if __name__ == "__main__":
    # Test the camera
    camera = RealSenseD455(width=1280, height=720, fps=30)
    
    if camera.initialize():
        print("Press 'q' to quit")
        
        while True:
            color, depth, depth_colormap, accel, gyro, timestamp = camera.get_frame()
            
            if color is not None and depth is not None:
                # Display images
                cv2.imshow('RGB', color)
                cv2.imshow('Depth', depth_colormap)
                
                # Print depth at center
                h, w = depth.shape
                center_depth = depth[h//2, w//2]
                print(f"Center depth: {center_depth:.3f}m", end='\r')
                
                if accel is not None:
                    print(f"Accel: {accel}", end='\r')
                
                if gyro is not None:
                    print(f"Gyro: {gyro}", end='\r')
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize camera")
