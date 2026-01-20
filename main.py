"""
RealSense D455 Visual SLAM - Main Entry Point
A simple Visual SLAM system using Intel RealSense D455 camera
"""

import argparse
import sys
import os
import cv2
import numpy as np
from datetime import datetime

from camera_capture import RealSenseD455
from visual_slam import VisualSLAM
from imu_odometry import IMUOdometry
from fusion_odometry import FusionOdometry


class RealSenseSLAMApp:
    """Main application class for RealSense D455 Visual SLAM"""
    
    def __init__(self, config):
        """
        Initialize the SLAM application
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.camera = None
        self.slam = None
        self.imu_odom = None
        self.fusion_odom = None
        self.running = False
        self.output_dir = config.get('output_dir', 'output')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Trajectory data
        self.trajectory = []
        self.timestamps = []
        
    def initialize(self):
        """Initialize camera and SLAM system"""
        print("=" * 60)
        print("RealSense D455 Visual SLAM")
        print("=" * 60)
        
        # Initialize camera
        print("\n[1/2] Initializing RealSense D455 camera...")
        self.camera = RealSenseD455(
            width=self.config.get('width', 640),
            height=self.config.get('height', 480),
            fps=self.config.get('fps', 30)
        )
        
        if not self.camera.initialize():
            print("ERROR: Failed to initialize camera!")
            return False
        
        # Get camera intrinsics
        intrinsics = self.camera.get_camera_intrinsics()
        if intrinsics is None:
            print("ERROR: Failed to get camera intrinsics!")
            return False
        
        print(f"   Camera intrinsics:")
        # print(f"   - fx: {intrinsics['fx']:.2f}")
        # print(f"   - fy: {intrinsics['fy']:.2f}")
        # print(f"   - cx: {intrinsics['cx']:.2f}")
        # print(f"   - cy: {intrinsics['cy']:.2f}")
        # print(f"   - depth_scale: {intrinsics['depth_scale']:.6f}")
        print(intrinsics)
        
        # Initialize SLAM
        print("\n[2/2] Initializing Visual SLAM system...")
        self.slam = VisualSLAM(intrinsics)
        self.imu_odom = IMUOdometry()
        self.fusion_odom = FusionOdometry()
        
        print("\nInitialization complete!")
        print("=" * 60)
        return True
    
    def run(self):
        """Run the SLAM system"""
        self.running = True
        
        print("\nControls:")
        print("  [q] - Quit and save results")
        print("  [s] - Save trajectory")
        print("  [k] - Save current keyframe")
        print("  [r] - Reset SLAM")
        print("  [d] - Toggle depth display")
        print("=" * 60)
        
        show_depth = False
        frame_count = 0
        start_time = datetime.now()
        
        while self.running:
            # Get frame from camera
            color, depth, depth_colormap, accel, gyro, timestamp = self.camera.get_frame()
            
            if color is None or depth is None:
                print("Warning: Failed to get frame from camera")
                continue
            
            frame_count += 1
            
            # Process frame with SLAM
            slam_result = self.slam.process_frame(color, depth)

            # Process IMU Odometry
            imu_pos = self.imu_odom.process(accel, gyro, timestamp)
            
            # Process Fusion Odometry
            fused_pos = self.fusion_odom.process(slam_result, accel, gyro, timestamp)
            
            # Store trajectory
            if slam_result['tracking']:
                pose = slam_result['pose'][:3, 3]
                self.trajectory.append(pose)
                self.timestamps.append(datetime.now().timestamp() - start_time.timestamp())
            
            # Create visualization
            vis_frame = self.create_visualization(color, slam_result, frame_count, accel, gyro)
            traj_map = self.create_trajectory_map(self.trajectory, "Visual Path")
            
            # Create IMU trajectory map
            imu_traj = self.imu_odom.get_trajectory()
            imu_map = self.create_trajectory_map(imu_traj, "IMU Path (Drifting)")
            
            # Create Fusion trajectory map
            fused_traj = self.fusion_odom.get_trajectory()
            fused_map = self.create_trajectory_map(fused_traj, "Fused Path (Robust)")
            
            # Display depth if enabled
            if show_depth:
                cv2.imshow('Depth', depth_colormap)
            
            # Display main window
            cv2.imshow('RealSense D455 Visual SLAM', vis_frame)
            cv2.imshow('Visual Trajectory (XZ)', traj_map)
            cv2.imshow('IMU Odometry (XZ)', imu_map)
            cv2.imshow('Fused Odometry (XZ)', fused_map)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.save_trajectory()
            elif key == ord('k'):
                self.save_keyframe(color, frame_count)
            elif key == ord('r'):
                self.reset_slam()
            elif key == ord('d'):
                show_depth = not show_depth
                if not show_depth:
                    cv2.destroyWindow('Depth')
        
        # Cleanup
        self.cleanup()
    
    def create_visualization(self, frame, slam_result, frame_count, accel, gyro):
        """
        Create visualization frame with SLAM information
        
        Args:
            frame: RGB image
            slam_result: SLAM processing result
            frame_count: Current frame number
            accel: Accelerometer data
            gyro: Gyroscope data
            
        Returns:
            np.ndarray: Visualization frame
        """
        vis_frame = frame.copy()
        
        # Draw keypoints
        cv2.drawKeypoints(vis_frame, self.slam.prev_keypoints, vis_frame, 
                         color=(0, 255, 0), flags=0)
        
        # Add status bar
        if slam_result.get('stationary', False):
            status_color = (0, 165, 255)  # Orange for stationary
            status_text = "STATIONARY"
        elif slam_result['tracking']:
            status_color = (0, 255, 0)    # Green for tracking
            status_text = "TRACKING"
        else:
            status_color = (0, 0, 255)    # Red for lost
            status_text = "LOST"
            
        cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], 40), status_color, -1)
        cv2.putText(vis_frame, f"Status: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add statistics
        y_offset = 60
        line_height = 25
        
        stats = [
            f"Frame: {frame_count}",
            f"Features: {slam_result['num_features']}",
            f"Matches: {slam_result.get('num_matches', 0)}",
            f"Keyframes: {len(self.slam.keyframes)}",
            f"Map Points: {len(self.slam.map_points)}",
        ]
        
        for stat in stats:
            cv2.putText(vis_frame, stat, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
        
        # Add pose information
        if slam_result['tracking']:
            pose = slam_result['pose'][:3, 3]
            cv2.putText(vis_frame, f"Position:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            cv2.putText(vis_frame, f"  X: {pose[0]:.3f} m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            cv2.putText(vis_frame, f"  Y: {pose[1]:.3f} m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            cv2.putText(vis_frame, f"  Z: {pose[2]:.3f} m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height

        # Add IMU information
        if accel is not None or gyro is not None:
            y_offset += 10
            cv2.putText(vis_frame, "IMU Data:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
            
            if accel is not None:
                cv2.putText(vis_frame, f"  Accel: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}]", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += line_height
            
            if gyro is not None:
                cv2.putText(vis_frame, f"  Gyro:  [{gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f}]", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_frame
    
    def create_trajectory_map(self, trajectory, title="Trajectory"):
        """
        Create a 2D map of the trajectory (XZ plane - top down view)
        
        Args:
            trajectory: List or array of 3D points
            title: Title to display on the map
            
        Returns:
            np.ndarray: Trajectory map image
        """
        map_size = 500
        traj_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        if len(trajectory) < 2:
            return traj_map
            
        # Extract X and Z coordinates (top-down view)
        # Convert list of arrays to numpy array
        points = np.array(trajectory)
        if points.ndim > 1:
            x_coords = points[:, 0]
            z_coords = points[:, 2]
        else:
            # Handle case where trajectory might have single point or weird shape
            return traj_map
        
        # Determine scale and offset to fit trajectory in map
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_z, max_z = np.min(z_coords), np.max(z_coords)
        
        range_x = max_x - min_x
        range_z = max_z - min_z
        max_range = max(range_x, range_z, 1.0)  # Avoid division by zero
        
        # auto-scaling: 1 meter = scale pixels
        # Fill 80% of the map
        scale = (map_size * 0.8) / max_range
        
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        # Transform points to map coordinates
        # Map center is (map_size/2, map_size/2)
        map_x = (x_coords - center_x) * scale + map_size / 2
        map_z = (map_size / 2) - (z_coords - center_z) * scale  # Invert Z for display
        
        # Draw trajectory lines
        points_map = np.column_stack((map_x, map_z)).astype(np.int32)
        cv2.polylines(traj_map, [points_map], False, (0, 255, 255), 1)
            
        # Draw start and end points
        cv2.circle(traj_map, tuple(points_map[0]), 3, (0, 255, 0), -1)  # Start Green
        cv2.circle(traj_map, tuple(points_map[-1]), 4, (0, 0, 255), -1)  # End Red
        
        # Add info
        cv2.putText(traj_map, title, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(traj_map, f"Scale: {scale:.1f} px/m", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(traj_map, f"Range: X:{range_x:.1f}m Z:{range_z:.1f}m", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
        return traj_map

    def save_trajectory(self):
        """Save trajectory to file"""
        if len(self.trajectory) == 0:
            print("Warning: No trajectory data to save")
            return
        
        filename = os.path.join(self.output_dir, 'trajectory.txt')
        np.savetxt(filename, self.trajectory, fmt='%.6f', 
                   header='x y z', comments='')
        print(f"Trajectory saved to {filename}")
        
        # Also save with timestamps
        timestamp_filename = os.path.join(self.output_dir, 'trajectory_with_timestamps.txt')
        data = np.column_stack((self.timestamps, self.trajectory))
        np.savetxt(timestamp_filename, data, fmt='%.6f', 
                   header='timestamp x y z', comments='')
        print(f"Trajectory with timestamps saved to {timestamp_filename}")
    
    def save_keyframe(self, frame, frame_count):
        """Save current frame as keyframe"""
        filename = os.path.join(self.output_dir, f'keyframe_{frame_count:06d}.png')
        cv2.imwrite(filename, frame)
        print(f"Keyframe saved to {filename}")
    
    def reset_slam(self):
        """Reset SLAM system"""
        print("Resetting SLAM system...")
        self.slam = VisualSLAM(self.camera.get_camera_intrinsics())
        self.imu_odom = IMUOdometry()
        self.fusion_odom = FusionOdometry()
        self.trajectory = []
        self.timestamps = []
        print("SLAM system reset")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 60)
        print("Shutting down...")
        
        # Save final trajectory
        if len(self.trajectory) > 0:
            self.save_trajectory()
        
        # Stop camera
        if self.camera:
            self.camera.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("Shutdown complete!")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RealSense D455 Visual SLAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --width 1280 --height 720  # Higher resolution
  python main.py --fps 15           # Lower frame rate
  python main.py --output-dir my_output  # Custom output directory
        """
    )
    
    parser.add_argument('--width', type=int, default=1280,
                       help='Image width (default: 640)')
    parser.add_argument('--height', type=int, default=720,
                       help='Image height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'width': args.width,
        'height': args.height,
        'fps': args.fps,
        'output_dir': args.output_dir
    }
    
    # Create and run application
    app = RealSenseSLAMApp(config)
    
    if app.initialize():
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            app.cleanup()
            sys.exit(0)
    else:
        print("Failed to initialize application")
        sys.exit(1)


if __name__ == "__main__":
    main()
