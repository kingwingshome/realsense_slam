"""
Trajectory Replayer Module
Visualize and replay recorded SLAM trajectory data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from typing import Tuple, Optional, List
import json


class TrajectoryReplayer:
    """Replay and visualize SLAM trajectory data"""
    
    def __init__(self, trajectory_file: str, timestamps_file: str = None):
        """
        Initialize trajectory replayer
        
        Args:
            trajectory_file: Path to trajectory file (txt format)
            timestamps_file: Optional path to timestamps file
        """
        self.trajectory_file = trajectory_file
        self.timestamps_file = timestamps_file
        self.trajectory = None
        self.timestamps = None
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.fig = None
        self.ax = None
        self.line = None
        self.point = None
        self.animation = None
        
        # Load trajectory data
        self.load_data()
    
    def load_data(self):
        """Load trajectory data from file"""
        try:
            # Load trajectory
            try:
                self.trajectory = np.loadtxt(self.trajectory_file)
            except ValueError:
                print("Warning: Failed to load with default settings, trying to skip header...")
                self.trajectory = np.loadtxt(self.trajectory_file, skiprows=1)

            print(f"Loaded {len(self.trajectory)} trajectory points from {self.trajectory_file}")
            
            # Load timestamps if provided
            if self.timestamps_file and os.path.exists(self.timestamps_file):
                try:
                    data = np.loadtxt(self.timestamps_file)
                except ValueError:
                    data = np.loadtxt(self.timestamps_file, skiprows=1)
                
                self.timestamps = data[:, 0]
                self.trajectory = data[:, 1:]  # x, y, z
                print(f"Loaded {len(self.timestamps)} timestamps from {self.timestamps_file}")
            else:
                # Generate default timestamps
                self.timestamps = np.arange(len(self.trajectory)) / 30.0  # Assume 30 FPS
                print("Using default timestamps (assuming 30 FPS)")
            
            # Print trajectory statistics
            self.print_statistics()
            
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            raise
    
    def print_statistics(self):
        """Print trajectory statistics"""
        if self.trajectory is None:
            return
        
        print("\n" + "=" * 50)
        print("Trajectory Statistics")
        print("=" * 50)
        print(f"Number of points: {len(self.trajectory)}")
        print(f"Duration: {self.timestamps[-1]:.2f} seconds")
        print(f"Average FPS: {len(self.trajectory) / self.timestamps[-1]:.2f}")
        
        # Calculate path length
        diffs = np.diff(self.trajectory, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        path_length = np.sum(distances)
        print(f"Total path length: {path_length:.3f} meters")
        
        # Calculate bounding box
        min_pos = np.min(self.trajectory, axis=0)
        max_pos = np.max(self.trajectory, axis=0)
        print(f"Bounding box:")
        print(f"  X: [{min_pos[0]:.3f}, {max_pos[0]:.3f}] m")
        print(f"  Y: [{min_pos[1]:.3f}, {max_pos[1]:.3f}] m")
        print(f"  Z: [{min_pos[2]:.3f}, {max_pos[2]:.3f}] m")
        
        # Calculate displacement
        start_pos = self.trajectory[0]
        end_pos = self.trajectory[-1]
        displacement = np.linalg.norm(end_pos - start_pos)
        print(f"Net displacement: {displacement:.3f} meters")
        print("=" * 50 + "\n")
    
    def plot_trajectory(self, save_path: str = None):
        """
        Plot static trajectory
        
        Args:
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], 
                'b-', linewidth=2, label='Trajectory')
        
        # Mark start and end
        ax.scatter(self.trajectory[0, 0], self.trajectory[0, 1], self.trajectory[0, 2], 
                  c='g', s=100, marker='o', label='Start')
        ax.scatter(self.trajectory[-1, 0], self.trajectory[-1, 1], self.trajectory[-1, 2], 
                  c='r', s=100, marker='s', label='End')
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('SLAM Trajectory', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Equal aspect ratio
        self._set_equal_aspect(ax)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_2d_projections(self, save_path: str = None):
        """
        Plot 2D projections of trajectory
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # XY projection
        axes[0].plot(self.trajectory[:, 0], self.trajectory[:, 1], 'b-', linewidth=2)
        axes[0].scatter(self.trajectory[0, 0], self.trajectory[0, 1], c='g', s=100, marker='o', label='Start')
        axes[0].scatter(self.trajectory[-1, 0], self.trajectory[-1, 1], c='r', s=100, marker='s', label='End')
        axes[0].set_xlabel('X (m)', fontsize=12)
        axes[0].set_ylabel('Y (m)', fontsize=12)
        axes[0].set_title('XY Projection (Top View)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # XZ projection
        axes[1].plot(self.trajectory[:, 0], self.trajectory[:, 2], 'b-', linewidth=2)
        axes[1].scatter(self.trajectory[0, 0], self.trajectory[0, 2], c='g', s=100, marker='o', label='Start')
        axes[1].scatter(self.trajectory[-1, 0], self.trajectory[-1, 2], c='r', s=100, marker='s', label='End')
        axes[1].set_xlabel('X (m)', fontsize=12)
        axes[1].set_ylabel('Z (m)', fontsize=12)
        axes[1].set_title('XZ Projection (Side View)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        # YZ projection
        axes[2].plot(self.trajectory[:, 1], self.trajectory[:, 2], 'b-', linewidth=2)
        axes[2].scatter(self.trajectory[0, 1], self.trajectory[0, 2], c='g', s=100, marker='o', label='Start')
        axes[2].scatter(self.trajectory[-1, 1], self.trajectory[-1, 2], c='r', s=100, marker='s', label='End')
        axes[2].set_xlabel('Y (m)', fontsize=12)
        axes[2].set_ylabel('Z (m)', fontsize=12)
        axes[2].set_title('YZ Projection (Front View)', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D projections saved to {save_path}")
        
        plt.show()
    
    def animate_trajectory(self, interval: int = 50, save_path: str = None):
        """
        Animate trajectory playback
        
        Args:
            interval: Animation interval in milliseconds
            save_path: Optional path to save animation as GIF
        """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.point, = self.ax.plot([], [], [], 'ro', markersize=10, label='Current Position')
        
        # Set labels and title
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.set_title('Trajectory Replay', fontsize=14, fontweight='bold')
        self.ax.legend()
        
        # Set axis limits
        self._set_axis_limits()
        self._set_equal_aspect(self.ax)
        
        # Add info text
        self.info_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, 
                                       fontsize=12, verticalalignment='top')
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, self._update_animation, frames=len(self.trajectory),
            interval=interval, blit=False, repeat=True
        )
        
        print("Starting trajectory animation...")
        print("Close the window to stop animation")
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            self.animation.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to {save_path}")
    
    def _update_animation(self, frame):
        """Update animation frame"""
        # Update trajectory line
        self.line.set_data(self.trajectory[:frame+1, 0], self.trajectory[:frame+1, 1])
        self.line.set_3d_properties(self.trajectory[:frame+1, 2])
        
        # Update current position point
        self.point.set_data([self.trajectory[frame, 0]], [self.trajectory[frame, 1]])
        self.point.set_3d_properties([self.trajectory[frame, 2]])
        
        # Update info text
        pos = self.trajectory[frame]
        time = self.timestamps[frame]
        self.info_text.set_text(
            f'Frame: {frame}/{len(self.trajectory)-1}\n'
            f'Time: {time:.2f} s\n'
            f'Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m'
        )
        
        return self.line, self.point, self.info_text
    
    def _set_axis_limits(self):
        """Set axis limits based on trajectory bounds"""
        min_pos = np.min(self.trajectory, axis=0)
        max_pos = np.max(self.trajectory, axis=0)
        
        # Add padding
        padding = 0.5
        self.ax.set_xlim(min_pos[0] - padding, max_pos[0] + padding)
        self.ax.set_ylim(min_pos[1] - padding, max_pos[1] + padding)
        self.ax.set_zlim(min_pos[2] - padding, max_pos[2] + padding)
    
    def _set_equal_aspect(self, ax):
        """Set equal aspect ratio for 3D plot"""
        try:
            # For matplotlib 3.3+
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # For older matplotlib versions
            pass
    
    def export_to_json(self, output_path: str):
        """
        Export trajectory to JSON format
        
        Args:
            output_path: Output JSON file path
        """
        data = {
            'trajectory': self.trajectory.tolist(),
            'timestamps': self.timestamps.tolist(),
            'metadata': {
                'num_points': len(self.trajectory),
                'duration': float(self.timestamps[-1]),
                'path_length': float(np.sum(np.sqrt(np.sum(np.diff(self.trajectory, axis=0)**2, axis=1))))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Trajectory exported to {output_path}")
    
    def export_to_kitti(self, output_path: str):
        """
        Export trajectory to KITTI format
        
        Args:
            output_path: Output file path
        """
        # KITTI format: timestamp tx ty tz qx qy qz qw
        # For simplicity, we only export position (no rotation)
        with open(output_path, 'w') as f:
            for i in range(len(self.trajectory)):
                timestamp = self.timestamps[i]
                x, y, z = self.trajectory[i]
                f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} 0.000000 0.000000 0.000000 1.000000\n")
        
        print(f"Trajectory exported to KITTI format: {output_path}")
    
    def export_to_tum(self, output_path: str):
        """
        Export trajectory to TUM format
        
        Args:
            output_path: Output file path
        """
        # TUM format: timestamp tx ty tz qx qy qz qw
        with open(output_path, 'w') as f:
            for i in range(len(self.trajectory)):
                timestamp = self.timestamps[i]
                x, y, z = self.trajectory[i]
                f.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} 0.000000 0.000000 0.000000 1.000000\n")
        
        print(f"Trajectory exported to TUM format: {output_path}")


def main():
    """Main entry point for trajectory replayer"""
    parser = argparse.ArgumentParser(
        description='SLAM Trajectory Replayer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trajectory_replayer.py trajectory.txt                    # Static plot
  python trajectory_replayer.py trajectory.txt --animate         # Animate trajectory
  python trajectory_replayer.py trajectory.txt --2d              # 2D projections
  python trajectory_replayer.py trajectory.txt --export-json output.json
  python trajectory_replayer.py trajectory.txt --timestamps trajectory_with_timestamps.txt
        """
    )
    
    parser.add_argument('trajectory_file', type=str,
                       help='Path to trajectory file (txt format)')
    parser.add_argument('--timestamps', type=str, default=None,
                       help='Path to timestamps file (optional)')
    parser.add_argument('--animate', action='store_true',
                       help='Animate trajectory playback')
    parser.add_argument('--2d', action='store_true', dest='plot_2d',
                       help='Show 2D projections')
    parser.add_argument('--interval', type=int, default=50,
                       help='Animation interval in milliseconds (default: 50)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save static plot')
    parser.add_argument('--save-animation', type=str, default=None,
                       help='Path to save animation (GIF format)')
    parser.add_argument('--export-json', type=str, default=None,
                       help='Export trajectory to JSON format')
    parser.add_argument('--export-kitti', type=str, default=None,
                       help='Export trajectory to KITTI format')
    parser.add_argument('--export-tum', type=str, default=None,
                       help='Export trajectory to TUM format')
    
    args = parser.parse_args()
    
    # Check if trajectory file exists
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file '{args.trajectory_file}' not found")
        return
    
    # Create replayer
    replayer = TrajectoryReplayer(args.trajectory_file, args.timestamps)
    
    # Handle different modes
    if args.animate:
        replayer.animate_trajectory(interval=args.interval, save_path=args.save_animation)
    elif args.export_json:
        replayer.export_to_json(args.export_json)
    elif args.export_kitti:
        replayer.export_to_kitti(args.export_kitti)
    elif args.export_tum:
        replayer.export_to_tum(args.export_tum)
    elif args.plot_2d:
        replayer.plot_2d_projections(save_path=args.save_plot)
    else:
        replayer.plot_trajectory(save_path=args.save_plot)


if __name__ == "__main__":
    main()
