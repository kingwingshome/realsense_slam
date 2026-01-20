import numpy as np
import cv2

class IMUOdometry:
    """
    Pure IMU Odometry (Dead Reckoning)
    Estimates position and orientation by integrating IMU data.
    Note: Pure IMU integration drifts very quickly without external corrections.
    """
    
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        # Rotation matrix from Body to World (Input is Body)
        self.R = np.eye(3)
        
        self.prev_timestamp = None
        self.initialized = False
        
        # Gravity vector in World Frame (estimated at start)
        self.gravity = None
        self.init_buffer = []
        self.init_frames = 30  # Number of frames to estimate gravity
        
        self.trajectory = []
        
    def process(self, accel, gyro, timestamp):
        """
        Process new IMU measurement
        
        Args:
            accel: Accelerometer data [ax, ay, az] in m/s^2
            gyro: Gyroscope data [gx, gy, gz] in rad/s
            timestamp: Current timestamp in seconds
            
        Returns:
            np.ndarray: Current position [x, y, z] or None if calibrating
        """
        if accel is None or gyro is None:
            return None
            
        # Initialization phase (Estimate Gravity)
        if not self.initialized:
            self.init_buffer.append(accel)
            if len(self.init_buffer) >= self.init_frames:
                # Average acceleration is gravity (assuming stationary start)
                self.gravity = np.mean(self.init_buffer, axis=0)
                norm = np.linalg.norm(self.gravity)
                print(f"IMU Odom Initialized. Gravity magnitude: {norm:.3f}")
                self.initialized = True
                self.prev_timestamp = timestamp
            return None

        # Time delta
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return None
            
        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        
        if dt > 0.1: # Ignore large gaps
            return self.position

        # 1. Update Orientation (Gyro Integration)
        # Create rotation delta from gyro
        r_vec = gyro * dt
        R_delta, _ = cv2.Rodrigues(r_vec)
        
        # Update orientation: R_new = R_old * R_delta
        self.R = self.R @ R_delta
        
        # 2. Update Position (Accel Integration)
        # Transform accel to World Frame
        accel_world = self.R @ accel
        
        # Remove gravity
        accel_net = accel_world - self.gravity
        
        # Integrate Velocity: v = v + a * dt
        self.velocity += accel_net * dt
        
        # Integrate Position: p = p + v * dt + 0.5 * a * dt^2
        self.position += self.velocity * dt + 0.5 * accel_net * (dt**2)
        
        self.trajectory.append(self.position.copy())
        
        return self.position

    def get_trajectory(self):
        return np.array(self.trajectory)
