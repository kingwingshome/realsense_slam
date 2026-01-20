"""
Visual SLAM Module
Implements a simple feature-based Visual SLAM using ORB features
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import time


class VisualSLAM:
    """Simple Visual SLAM implementation using ORB features and pose estimation"""
    
    def __init__(self, camera_intrinsics: dict, vocab_path: str = None):
        """
        Initialize Visual SLAM
        
        Args:
            camera_intrinsics: Camera intrinsic parameters
            vocab_path: Path to vocabulary file (optional)
        """
        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']
        self.width = camera_intrinsics['width']
        self.height = camera_intrinsics['height']
        
        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion for RealSense)
        self.D = np.zeros((5, 1), dtype=np.float32)
        
        # ORB feature detector
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # BFMatcher for feature matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # State variables
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.pose = np.eye(4)  # Initial pose (identity matrix)
        self.keyframes = []
        self.map_points = []
        self.frame_count = 0
        
        # Tracking state
        self.is_tracking = False
        
        print("Visual SLAM initialized")
    
    def extract_features(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features from frame
        
        Args:
            frame: Input image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.7) -> List:
        """
        Match features between two descriptor sets using Lowe's ratio test
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            ratio_threshold: Ratio threshold for Lowe's test
            
        Returns:
            list: List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Match descriptors
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_pose(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        """
        Estimate camera pose from matched features
        
        Args:
            kp1: Keypoints from previous frame
            kp2: Keypoints from current frame
            matches: Matched feature pairs
            
        Returns:
            np.ndarray: 4x4 transformation matrix or None if estimation fails
        """
        if len(matches) < 10:
            return None
        
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, 
                                      prob=0.999, threshold=1.0)
        
        if E is None:
            return None
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T
    
    def triangulate_points(self, kp1: List, kp2: List, matches: List, 
                          T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from matched features
        
        Args:
            kp1: Keypoints from previous frame
            kp2: Keypoints from current frame
            matches: Matched feature pairs
            T1: Pose of first camera
            T2: Pose of second camera
            
        Returns:
            np.ndarray: 3D points
        """
        if len(matches) < 10:
            return np.array([])
        
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Projection matrices
        P1 = self.K @ T1[:3, :]
        P2 = self.K @ T2[:3, :]
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def check_for_motion(self, kp1: List, kp2: List, matches: List, threshold: float = 1.0) -> bool:
        """
        Check if there is significant motion based on average feature displacement
        
        Args:
            kp1: Keypoints from previous frame
            kp2: Keypoints from current frame
            matches: Matched feature pairs
            threshold: Pixel movement threshold (default: 1.0 pixel)
            
        Returns:
            bool: True if motion is detected
        """
        if not matches:
            return False
            
        # varied motion calculation
        displacements = []
        for m in matches:
            pt1 = np.array(kp1[m.queryIdx].pt)
            pt2 = np.array(kp2[m.trainIdx].pt)
            dist = np.linalg.norm(pt2 - pt1)
            displacements.append(dist)
            
        avg_disp = np.mean(displacements)
        return avg_disp > threshold

    def process_frame(self, frame: np.ndarray, depth: np.ndarray = None) -> dict:
        """
        Process a new frame and update SLAM state
        
        Args:
            frame: RGB image
            depth: Depth image (optional, for RGB-D SLAM)
            
        Returns:
            dict: SLAM state information
        """
        self.frame_count += 1
        
        # Extract features
        keypoints, descriptors = self.extract_features(frame)
        
        result = {
            'frame_count': self.frame_count,
            'num_features': len(keypoints),
            'pose': self.pose.copy(),
            'tracking': False,
            'keyframe_added': False,
            'stationary': False
        }
        
        # First frame - initialize
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.is_tracking = True
            result['tracking'] = True
            print(f"Frame {self.frame_count}: Initialized with {len(keypoints)} features")
            return result
        
        # Match features
        matches = self.match_features(self.prev_descriptors, descriptors)
        result['num_matches'] = len(matches)
        
        if len(matches) < 10:
            print(f"Frame {self.frame_count}: Lost tracking (only {len(matches)} matches)")
            self.is_tracking = False
            return result
            
        # [Fix] Check for significant motion before estimating pose
        # If average feature movement is < 0.5 pixels, assume stationary
        if not self.check_for_motion(self.prev_keypoints, keypoints, matches, threshold=0.5):
            # Camera is stationary
            result['tracking'] = True
            result['stationary'] = True
            # Update previous frame data but KEEP current pose
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return result
        
        # Estimate pose
        T = self.estimate_pose(self.prev_keypoints, keypoints, matches)
        
        if T is not None:
            # [Fix] Filter out very small translations (noise)
            translation_norm = np.linalg.norm(T[:3, 3])
            
            # Threshold: 1mm movement
            if translation_norm < 0.001:
                result['tracking'] = True
                result['stationary'] = True
                self.prev_frame = frame.copy()
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                return result

            # Update pose
            self.pose = self.pose @ np.linalg.inv(T)
            result['pose'] = self.pose.copy()
            result['tracking'] = True
            self.is_tracking = True
            
            # Triangulate new map points
            if depth is not None:
                # RGB-D: use depth directly
                points_3d = self.depth_to_3d(keypoints, depth, matches)
            else:
                # Visual only: triangulate
                points_3d = self.triangulate_points(
                    self.prev_keypoints, keypoints, matches,
                    np.eye(4), T
                )
            
            if len(points_3d) > 0:
                self.map_points.extend(points_3d)
            
            # Check if keyframe should be added
            if self.should_add_keyframe(len(matches)):
                self.add_keyframe(frame, keypoints, descriptors, self.pose.copy())
                result['keyframe_added'] = True
                print(f"Frame {self.frame_count}: Keyframe added ({len(self.keyframes)} total)")
            
            # Reduce logging frequency
            if self.frame_count % 30 == 0:
                 print(f"Frame {self.frame_count}: Tracking ({len(matches)} matches, "
                       f"pos: {self.pose[:3, 3].round(3)})")
        else:
            print(f"Frame {self.frame_count}: Pose estimation failed")
            self.is_tracking = False
        
        # Update previous frame
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return result
    
    def depth_to_3d(self, keypoints: List, depth: np.ndarray, 
                   matches: List) -> np.ndarray:
        """
        Convert 2D keypoints to 3D points using depth information
        
        Args:
            keypoints: Keypoints from current frame
            depth: Depth image
            matches: Matched features
            
        Returns:
            np.ndarray: 3D points
        """
        points_3d = []
        
        for m in matches:
            pt = keypoints[m.trainIdx].pt
            x, y = int(pt[0]), int(pt[1])
            
            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                d = depth[y, x]
                if d > 0:
                    # Convert to 3D
                    z = d
                    x_3d = (x - self.cx) * z / self.fx
                    y_3d = (y - self.cy) * z / self.fy
                    points_3d.append([x_3d, y_3d, z])
        
        return np.array(points_3d) if points_3d else np.array([])
    
    def should_add_keyframe(self, num_matches: int, threshold: int = 100) -> bool:
        """
        Determine if a new keyframe should be added
        
        Args:
            num_matches: Number of feature matches
            threshold: Threshold for adding keyframe
            
        Returns:
            bool: True if keyframe should be added
        """
        # Add keyframe if matches are below threshold (low overlap)
        # or if it's the first keyframe
        if len(self.keyframes) == 0:
            return True
        return num_matches < threshold
    
    def add_keyframe(self, frame: np.ndarray, keypoints: List, 
                    descriptors: np.ndarray, pose: np.ndarray):
        """
        Add a keyframe to the database
        
        Args:
            frame: Frame image
            keypoints: Keypoints
            descriptors: Feature descriptors
            pose: Camera pose
        """
        self.keyframes.append({
            'frame': frame.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy()
        })
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get the camera trajectory
        
        Returns:
            np.ndarray: Array of camera positions
        """
        trajectory = []
        for kf in self.keyframes:
            trajectory.append(kf['pose'][:3, 3])
        return np.array(trajectory)
    
    def save_trajectory(self, filename: str):
        """
        Save trajectory to file
        
        Args:
            filename: Output filename
        """
        trajectory = self.get_trajectory()
        np.savetxt(filename, trajectory, fmt='%.6f')
        print(f"Trajectory saved to {filename}")
    
    def visualize_frame(self, frame: np.ndarray, matches: List = None) -> np.ndarray:
        """
        Visualize features and matches on frame
        
        Args:
            frame: Input frame
            matches: Optional matches to visualize
            
        Returns:
            np.ndarray: Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw keypoints
        cv2.drawKeypoints(vis_frame, self.prev_keypoints, vis_frame, 
                         color=(0, 255, 0), flags=0)
        
        return vis_frame


if __name__ == "__main__":
    # Test Visual SLAM
    from camera_capture import RealSenseD455
    
    # Initialize camera
    camera = RealSenseD455(width=640, height=480, fps=30)
    
    if camera.initialize():
        # Get camera intrinsics
        intrinsics = camera.get_camera_intrinsics()
        
        # Initialize SLAM
        slam = VisualSLAM(intrinsics)
        
        print("Press 'q' to quit, 's' to save trajectory")
        
        while True:
            color, depth, _ = camera.get_frame()
            
            if color is not None:
                # Process frame with SLAM
                result = slam.process_frame(color, depth)
                
                # Visualize
                vis_frame = slam.visualize_frame(color)
                
                # Display info
                cv2.putText(vis_frame, f"Features: {result['num_features']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Tracking: {'Yes' if result['tracking'] else 'No'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Keyframes: {len(slam.keyframes)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display pose
                pose = result['pose'][:3, 3]
                cv2.putText(vis_frame, f"Pos: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Visual SLAM', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    slam.save_trajectory('trajectory.txt')
        
        # Save trajectory before exit
        slam.save_trajectory('trajectory.txt')
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize camera")
