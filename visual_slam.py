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
        self.prev_points_3d = None  # 3D points of previous frame
        self.velocity = np.eye(4)   # Camera velocity (T_current_prev)
        self.last_T = np.eye(4)     # Last relative motion
        
        print("Visual SLAM initialized")
    
    def extract_features(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features with uniform distribution (Grid-based selection)
        """
        # Strategy: Detect many features globally, then select best ones per grid cell
        
        rows, cols = 4, 4
        h, w = frame.shape[:2]
        h_cell, w_cell = h // rows, w // cols
        
        target_total = 2000
        n_features_per_cell = int(target_total / (rows * cols))
        
        # 1. Detect a large pool of features using a temporary heavy detector
        pool_orb = cv2.ORB_create(
            nfeatures=target_total * 3,  # Detect 3x detailed pool
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Detect keypoints only first (faster)
        keypoints = pool_orb.detect(frame, None)
        
        if not keypoints:
            return [], None
            
        # 2. Bucket features into grid
        grid = [[[] for _ in range(cols)] for _ in range(rows)]
        
        for idx, kp in enumerate(keypoints):
            r = int(min(max(kp.pt[1], 0) // h_cell, rows - 1))
            c = int(min(max(kp.pt[0], 0) // w_cell, cols - 1))
            grid[r][c].append(kp)
            
        # 3. Select best features per cell
        final_keypoints = []
        for r in range(rows):
            for c in range(cols):
                cell_kps = grid[r][c]
                # Sort by response (strongest first)
                cell_kps.sort(key=lambda k: k.response, reverse=True)
                # Take top N
                final_keypoints.extend(cell_kps[:n_features_per_cell])
        
        # 4. Compute descriptors for selected keypoints
        final_keypoints, descriptors = self.orb.compute(frame, final_keypoints)
        
        return final_keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      kp1: List = None, kp2: List = None,
                      ratio_threshold: float = 0.75) -> List:
        """
        Match features between two descriptor sets using Lowe's ratio test
        and optionally Orientation Consistency Check (like ORB-SLAM)
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            kp1: First set of keypoints (optional, for rotation check)
            kp2: Second set of keypoints (optional, for rotation check)
            ratio_threshold: Ratio threshold for Lowe's test
            
        Returns:
            list: List of good matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
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
                    
        # Orientation Consistency Check (ORB-SLAM trick)
        # Filters outliers by checking if feature rotation is consistent
        if kp1 is not None and kp2 is not None and len(good_matches) > 10:
            rotations = []
            valid_matches = []
            
            for m in good_matches:
                # angle is in degrees [0, 360)
                ang1 = kp1[m.queryIdx].angle
                ang2 = kp2[m.trainIdx].angle
                
                # If angle is valid
                if ang1 >= 0 and ang2 >= 0:
                    diff = (ang1 - ang2) % 360
                    rotations.append(diff)
                    valid_matches.append(m)
                else:
                    # Keep if no angle info (shouldn't happen with ORB)
                    valid_matches.append(m)
            
            if len(rotations) > 0:
                # Create histogram of rotations (30 bins = 12 degrees per bin)
                hist, bin_edges = np.histogram(rotations, bins=30, range=(0, 360))
                
                # Find top 3 bins
                top_indices = np.argsort(hist)[-3:]
                
                # Filter matches that belong to top bins
                final_matches = []
                bin_width = 360.0 / 30.0
                
                for m in valid_matches:
                    ang1 = kp1[m.queryIdx].angle
                    ang2 = kp2[m.trainIdx].angle
                    
                    if ang1 < 0 or ang2 < 0:
                        final_matches.append(m)
                        continue
                        
                    diff = (ang1 - ang2) % 360
                    bin_idx = int(diff / bin_width)
                    if bin_idx == 30: bin_idx = 29
                    
                    if bin_idx in top_indices:
                        final_matches.append(m)
                        
                return final_matches
            
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

    def estimate_pose_pnp(self, points_3d: np.ndarray, current_kp: List, matches: List, 
                         initial_guess: Tuple[np.ndarray, np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Estimate pose using PnP (3D from prev frame -> 2D in current frame)
        Inspired by ORB-SLAM3 tracking
        
        Args:
            points_3d: 3D points corresponding to previous keypoints (N, 3)
            current_kp: Keypoints from current frame
            matches: Matches between prev and current descriptors
            initial_guess: Tuple (rvec, tvec) for initialization
            
        Returns:
            np.ndarray: 4x4 transformation matrix (T_curr_prev) where P_curr = T * P_prev
        """
        if len(matches) < 10 or points_3d is None:
            return None
            
        # Get 3D points and 2D points pairs
        object_points = []
        image_points = []
        
        for m in matches:
            query_idx = m.queryIdx
            train_idx = m.trainIdx
            
            if query_idx < len(points_3d):
                p3d = points_3d[query_idx]
                if np.all(np.isfinite(p3d)):
                     object_points.append(p3d)
                     image_points.append(current_kp[train_idx].pt)
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        if len(object_points) < 6:
            return None
            
        flags = cv2.SOLVEPNP_ITERATIVE
        
        # Proper handling of solvePnPRansac arguments
        # OpenCV python wrappers can be finicky with optional arguments
        if initial_guess is not None:
            rvec_init, tvec_init = initial_guess
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, self.D,
                rvec=rvec_init, tvec=tvec_init,
                useExtrinsicGuess=True,
                iterationsCount=100, 
                reprojectionError=3.0,
                confidence=0.99,
                flags=flags
            )
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, self.D,
                useExtrinsicGuess=False,
                iterationsCount=100, 
                reprojectionError=3.0,
                confidence=0.99,
                flags=flags
            )
        
        if not success or inliers is None or len(inliers) < 10:
            return None

        # Refine pose using Levenberg-Marquardt optimization on inliers
        # This acts as the "Pose Optimization" step in ORB-SLAM
        try:
            inliers_idx = inliers.flatten()
            obj_inliers = object_points[inliers_idx]
            img_inliers = image_points[inliers_idx]
            
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_inliers, img_inliers, self.K, self.D, rvec, tvec
            )
        except Exception:
            # Fallback to RANSAC result if refinement fails
            pass

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T

    def track_reference_keyframe(self, current_kp: List, current_desc: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: Try to track against the last available KeyFrame
        Useful when frame-to-frame tracking fails (e.g. motion blur, occlusion)
        """
        if len(self.keyframes) == 0:
            return None
            
        ref_kf = self.keyframes[-1]
        
        matches = self.match_features(
            ref_kf['descriptors'], current_desc, 
            ref_kf['keypoints'], current_kp, # Pass keypoints for orientation check
            ratio_threshold=0.8
        )
        
        if len(matches) < 15:
            return None
            
        if 'points_3d' not in ref_kf or ref_kf['points_3d'] is None:
            return None
            
        T_curr_kf = self.estimate_pose_pnp(ref_kf['points_3d'], current_kp, matches)
        
        if T_curr_kf is not None:
             T_kf_curr = np.linalg.inv(T_curr_kf)
             new_pose = ref_kf['pose'] @ T_kf_curr
             return new_pose
             
        return None

    def compute_all_3d_points(self, keypoints: List, depth: np.ndarray) -> np.ndarray:
        """
        Compute 3D points for all keypoints using depth map
        """
        points_3d = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
        height, width = depth.shape
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            if 0 <= x < width and 0 <= y < height:
                d = depth[y, x]
                if d > 0.1 and d < 10.0:  
                    z = float(d)
                    x_3d = (x - self.cx) * z / self.fx
                    y_3d = (y - self.cy) * z / self.fy
                    points_3d[i] = [x_3d, y_3d, z]
                    
        return points_3d
    
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
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        P1 = self.K @ T1[:3, :]
        P2 = self.K @ T2[:3, :]
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        
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
        
        keypoints, descriptors = self.extract_features(frame)
        
        result = {
            'frame_count': self.frame_count,
            'num_features': len(keypoints),
            'pose': self.pose.copy(),
            'tracking': False,
            'keyframe_added': False,
            'stationary': False,
            'mode': 'RGB-D' if depth is not None else 'Monocular'
        }
        
        current_points_3d = None
        if depth is not None:
             current_points_3d = self.compute_all_3d_points(keypoints, depth)

        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_points_3d = current_points_3d
            self.is_tracking = True
            result['tracking'] = True
            print(f"Frame {self.frame_count}: Initialized with {len(keypoints)} features")
            return result
        
        matches = self.match_features(
            self.prev_descriptors, descriptors,
            self.prev_keypoints, keypoints # Pass keypoints for orientation check
        )
        result['num_matches'] = len(matches)
        
        if len(matches) < 15:
            print(f"Frame {self.frame_count}: Lost tracking (only {len(matches)} matches)")
            
            if len(self.keyframes) > 0:
                 print(f"  Attempting to track reference keyframe {len(self.keyframes)-1}...")
                 new_abs_pose = self.track_reference_keyframe(keypoints, descriptors)
                 if new_abs_pose is not None:
                     print(f"  Recovered tracking via Keyframe!")
                     self.pose = new_abs_pose
                     self.is_tracking = True
                     
                     self.prev_frame = frame.copy()
                     self.prev_keypoints = keypoints
                     self.prev_descriptors = descriptors
                     
                     if depth is not None:
                        self.prev_points_3d = self.compute_all_3d_points(keypoints, depth)
                     else:
                        self.prev_points_3d = None
                     
                     result['pose'] = self.pose.copy()
                     result['tracking'] = True
                     return result
            
            self.is_tracking = False
            return result
            
        if not self.check_for_motion(self.prev_keypoints, keypoints, matches, threshold=0.5):
            result['tracking'] = True
            result['stationary'] = True
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            if current_points_3d is not None:
                self.prev_points_3d = current_points_3d
            return result
        
        T = None
        
        guess = None
        if self.is_tracking and self.last_T is not None:
             guess_R = self.last_T[:3, :3]
             # Ensure translation is 3x1 vector
             guess_t = self.last_T[:3, 3].reshape(3, 1)
             guess_rvec, _ = cv2.Rodrigues(guess_R)
             guess = (guess_rvec, guess_t)
        
        if self.prev_points_3d is not None and depth is not None:
            T = self.estimate_pose_pnp(self.prev_points_3d, keypoints, matches, initial_guess=guess)
        
        if T is None and self.prev_points_3d is not None and depth is not None and guess is not None:
             T = self.estimate_pose_pnp(self.prev_points_3d, keypoints, matches, initial_guess=None)
            
        if T is None:
            T = self.estimate_pose(self.prev_keypoints, keypoints, matches)
        
        if T is None and len(self.keyframes) > 0:
             print(f"Frame {self.frame_count}: Local tracking failed, trying Keyframe...")
             new_abs_pose = self.track_reference_keyframe(keypoints, descriptors)
             if new_abs_pose is not None:
                  self.pose = new_abs_pose
                  result['pose'] = self.pose.copy()
                  result['tracking'] = True
                  self.is_tracking = True
                  self.prev_frame = frame.copy()
                  self.prev_keypoints = keypoints
                  self.prev_descriptors = descriptors
                  if current_points_3d is not None:
                      self.prev_points_3d = current_points_3d
                  return result

        if T is not None:
            # Filter out very small translations (noise)
            translation_norm = np.linalg.norm(T[:3, 3])
            
            # Calculate rotation angle
            R_rel = T[:3, :3]
            trace = np.trace(R_rel)
            trace = min(3.0, max(-1.0, trace))
            rotation_angle = np.arccos((trace - 1.0) / 2.0)
            
            # Threshold: 1mm movement or very small rotation
            if translation_norm < 0.001 and rotation_angle < 0.001:
                result['tracking'] = True
                result['stationary'] = True
                self.prev_frame = frame.copy()
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                if current_points_3d is not None:
                    self.prev_points_3d = current_points_3d
                return result

            self.last_T = T
            self.pose = self.pose @ np.linalg.inv(T)
            
            result['pose'] = self.pose.copy()
            result['tracking'] = True
            self.is_tracking = True
            
            if depth is not None:
                valid_indices = []
                for idx, pt in enumerate(current_points_3d):
                     if np.all(np.isfinite(pt)):
                          valid_indices.append(pt)
                
                if len(valid_indices) > 0:
                     self.map_points.extend(valid_indices[::10]) 
            else:
                points_3d = self.triangulate_points(
                    self.prev_keypoints, keypoints, matches,
                    np.eye(4), T
                )
                if len(points_3d) > 0:
                    self.map_points.extend(points_3d)
            
            # Keyframe selection
            # ORB-SLAM-like policy: Add keyframe if:
            # 1. Time/Frames passed (we don't strictly track time here, but handle via frame count implicitly by flow)
            # 2. Tracking is getting weak (few matches)
            # 3. Significant motion (Translation or Rotation)
            
            translation_dist = np.linalg.norm(T[:3, 3])
            
            # Conditions
            c1_weak_tracking = len(matches) < 200 # Need more points
            c2_motion = translation_dist > 0.05 or rotation_angle > 0.1 # Moved significantly
            
            # We enforce a minimum interval between keyframes to avoid flooding (e.g. 5 frames)
            # But if tracking is very weak, we add anyway.
            # In this simple loop, we don't have a frame counter since last keyframe readily available, 
            # but we can rely on significant motion logic.
            
            should_add = False
            
            if c1_weak_tracking and len(matches) > 50: # Weak but not failing
                should_add = True
            elif c2_motion:
                should_add = True
                
            # If we have very few keyframes, be generous
            if len(self.keyframes) < 5:
                should_add = True
            
            if should_add:
                # Pass current 3D points to keyframe for future tracking
                points_3d_to_save = current_points_3d if current_points_3d is not None else None
                self.add_keyframe(frame, keypoints, descriptors, self.pose.copy(), points_3d_to_save)
                
                result['keyframe_added'] = True
                print(f"Frame {self.frame_count}: Keyframe added ({len(self.keyframes)} total)")
            
            if self.frame_count % 30 == 0:
                 print(f"Frame {self.frame_count}: Tracking ({len(matches)} matches)")
        else:
            print(f"Frame {self.frame_count}: Pose estimation failed")
            self.is_tracking = False
        
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        if current_points_3d is not None:
            self.prev_points_3d = current_points_3d
        else:
            self.prev_points_3d = None
        
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
        if len(self.keyframes) == 0:
            return True
        return num_matches < threshold
    
    def add_keyframe(self, frame: np.ndarray, keypoints: List, 
                    descriptors: np.ndarray, pose: np.ndarray, points_3d: np.ndarray = None):
        """
        Add a keyframe to the database
        
        Args:
            frame: Frame image
            keypoints: Keypoints
            descriptors: Feature descriptors
            pose: Camera pose
            points_3d: Associated 3D points
        """
        self.keyframes.append({
            'frame': frame.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'points_3d': points_3d
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
        
        cv2.drawKeypoints(vis_frame, self.prev_keypoints, vis_frame, 
                         color=(0, 255, 0), flags=0)
        
        return vis_frame


if __name__ == "__main__":
    from camera_capture import RealSenseD455
    
    camera = RealSenseD455(width=640, height=480, fps=30)
    
    if camera.initialize():
        intrinsics = camera.get_camera_intrinsics()
        
        slam = VisualSLAM(intrinsics)
        
        print("Press 'q' to quit, 's' to save trajectory")
        
        while True:
            color, depth, _ = camera.get_frame()
            
            if color is not None:
                result = slam.process_frame(color, depth)
                
                vis_frame = slam.visualize_frame(color)
                
                cv2.putText(vis_frame, f"Features: {result['num_features']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Tracking: {'Yes' if result['tracking'] else 'No'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Keyframes: {len(slam.keyframes)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                pose = result['pose'][:3, 3]
                cv2.putText(vis_frame, f"Pos: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Visual SLAM', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    slam.save_trajectory('trajectory.txt')
        
        slam.save_trajectory('trajectory.txt')
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize camera")
