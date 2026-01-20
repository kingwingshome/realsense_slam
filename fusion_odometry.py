import numpy as np
import cv2

class FusionOdometry:
    """
    Visual-Inertial Fusion Odometry
    Combines Visual SLAM pose with IMU integration for robust tracking.
    Uses a simple loose coupling approach:
    - IMU provides high-frequency updates (prediction)
    - Visual SLAM provides low-frequency corrections (update)
    """
    
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        # Orientation (Rotation matrix from Body to World)
        self.R = np.eye(3)
        
        self.prev_timestamp = None
        self.initialized = False
        
        # Gravity vector (estimated during storage)
        self.gravity = None
        self.init_buffer = []
        self.init_frames = 30
        
        self.trajectory = []
        
        # Tuning parameters
        self.alpha_pos = 0.1  # Weight for Visual correction (0.0 to 1.0)
        # If Visual is available, how much do we trust it?
        # Actually, if Visual is tracking well, we should trust it almost 100% for position, 
        # but maybe filter it to be smooth. If we trust it 100%, it's not really fusion, just switching.
        # Let's try a complementary filter approach.
        
    def process(self, slam_result, accel, gyro, timestamp):
        """
        Process new data
        
        Args:
            slam_result: Dictionary from VisualSLAM
            accel: Accelerometer data
            gyro: Gyroscope data
            timestamp: Current timestamp
            
        Returns:
            np.ndarray: Fused position
        """
        # 1. Initialization (Gravity Estimation)
        if not self.initialized:
            if accel is not None:
                self.init_buffer.append(accel)
                if len(self.init_buffer) >= self.init_frames:
                    self.gravity = np.mean(self.init_buffer, axis=0)
                    self.initialized = True
                    self.prev_timestamp = timestamp
                    
                    # Initialize from SLAM if available
                    if slam_result['tracking']:
                        self.position = slam_result['pose'][:3, 3].copy()
                        self.R = slam_result['pose'][:3, :3].copy()
                        print(f"Fusion Initialized. Gravity: {np.linalg.norm(self.gravity):.2f}")
            return self.position

        # 2. Time update
        dt = 0
        if self.prev_timestamp is not None:
            dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        
        if dt > 0.1: # Reset if gap is too large
            dt = 0.0
            
        # 3. IMU Prediction (High Frequency / Dead Reckoning)
        if accel is not None and gyro is not None and dt > 0:
            # Gyro integration (Orientation)
            r_vec = gyro * dt
            R_delta, _ = cv2.Rodrigues(r_vec)
            self.R = self.R @ R_delta
            
            # Accel integration (Position/Velocity)
            accel_world = self.R @ accel
            accel_net = accel_world - self.gravity
            
            # Basic Euler integration
            self.position += self.velocity * dt + 0.5 * accel_net * (dt**2)
            self.velocity += accel_net * dt
            
        # 4. Visual Correction (Low Frequency)
        if slam_result['tracking']:
            visual_pos = slam_result['pose'][:3, 3]
            visual_R = slam_result['pose'][:3, :3]
            
            # Correct Position
            # We trust Visual SLAM for absolute position to stop drift
            # But we might want to smooth it
            
            # Strong correction to Visual SLAM (0.95) to prevent drift
            # but allow IMU to fill in smoothness
            alpha = 0.8 
            
            diff_pos = visual_pos - self.position
            # If difference is huge (SLAM jump), maybe just take SLAM
            if np.linalg.norm(diff_pos) > 0.5:
                 self.position = visual_pos.copy()
                 self.velocity = np.zeros(3) # Reset velocity on jump
            else:
                 self.position = self.position + alpha * diff_pos
            
            # Correct Orientation
            # Correcting rotation matrix is complex (averaging rotations)
            # Simple approach: Slerp or just reset if difference is small?
            # For simplicity: Reset to Visual Rotation
            self.R = visual_R
            
            # What about Velocity?
            # Visual SLAM doesn't give velocity. 
            # But if we correct position, we implicitly correct the path.
            # We don't directly correct velocity here which is substantial simplification.
            
        # Tracking logic for lost visual
        # If visual is lost, we rely purely on IMU (prediction step above)
        
        self.trajectory.append(self.position.copy())
        return self.position
        
    def get_trajectory(self):
        return np.array(self.trajectory)