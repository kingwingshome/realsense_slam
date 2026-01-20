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
        print(f"   - fx: {intrinsics['fx']:.2f}")
        print(f"   - fy: {intrinsics['fy']:.2f}")
        print(f"   - cx: {intrinsics['cx']:.2f}")
        print(f"   - cy: {intrinsics['cy']:.2f}")
        print(f"   - depth_scale: {intrinsics['depth_scale']:.6f}")
        
        # Initialize SLAM
        print("\n[2/2] Initializing Visual SLAM system...")
        self.slam = VisualSLAM(intrinsics)
        
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
            color, depth, depth_colormap = self.camera.get_frame()
            
            if color is None or depth is None:
                print("Warning: Failed to get frame from camera")
                continue
            
            frame_count += 1
            
            # Process frame with SLAM
            slam_result = self.slam.process_frame(color, depth)
            
            # Store trajectory
            if slam_result['tracking']:
                pose = slam_result['pose'][:3, 3]
                self.trajectory.append(pose)
                self.timestamps.append(datetime.now().timestamp() - start_time.timestamp())
            
            # Create visualization
            vis_frame = self.create_visualization(color, slam_result, frame_count)
            
            # Display depth if enabled
            if show_depth:
                cv2.imshow('Depth', depth_colormap)
            
            # Display main window
            cv2.imshow('RealSense D455 Visual SLAM', vis_frame)
            
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
    
    def create_visualization(self, frame, slam_result, frame_count):
        """
        Create visualization frame with SLAM information
        
        Args:
            frame: RGB image
            slam_result: SLAM processing result
            frame_count: Current frame number
            
        Returns:
            np.ndarray: Visualization frame
        """
        vis_frame = frame.copy()
        
        # Draw keypoints
        cv2.drawKeypoints(vis_frame, self.slam.prev_keypoints, vis_frame, 
                         color=(0, 255, 0), flags=0)
        
        # Add status bar
        status_color = (0, 255, 0) if slam_result['tracking'] else (0, 0, 255)
        status_text = "TRACKING" if slam_result['tracking'] else "LOST"
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
        
        return vis_frame
    
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
    
    parser.add_argument('--width', type=int, default=640,
                       help='Image width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
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
