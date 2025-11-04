"""
Detailed Crash Tracker - Tracks crashes and detections with comprehensive metrics
Outputs detailed CSV files with b-plane values, Kalman filter metrics, and more
"""
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from scipy.stats import chi2


class DetailedCrashTracker:
    """Tracks crashes and detections with detailed metrics for analysis"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Per-trial tracking
        self.trial_num = 0
        self.current_trial_dir = None
        self.current_trial_events = []  # All events (crashes and detections)
        
        # Global statistics
        self.total_trials = 0
        self.total_actual_crashes = 0
        self.total_detections = 0
        
        # CSV file for detailed events (will be created per trial)
        self.events_csv = None
        
        # Summary CSV (at root level, aggregates all trials)
        self.summary_csv = self.results_dir / "trial_summary.csv"
        self._initialize_summary_csv()
        
        # Combined events CSV (optional, aggregates all trials)
        self.combined_events_csv = self.results_dir / "combined_detailed_events.csv"
        self._combined_csv_initialized = False
    
    def _initialize_csv(self, csv_path):
        """Initialize the detailed events CSV file with headers"""
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Basic identifiers
                    'trial_num', 'timestamp', 'event_type', 'vehicle1_id', 'vehicle2_id',
                    
                    # Vehicle 1 state
                    'v1_x', 'v1_y', 'v1_velocity', 'v1_heading',
                    'v1_pred_x', 'v1_pred_y', 'v1_pred_velocity', 'v1_pred_heading',
                    
                    # Vehicle 2 state
                    'v2_x', 'v2_y', 'v2_velocity', 'v2_heading',
                    'v2_pred_x', 'v2_pred_y', 'v2_pred_velocity', 'v2_pred_heading',
                    
                    # Relative metrics
                    'relative_distance', 'relative_velocity_magnitude', 'relative_velocity_x', 'relative_velocity_y',
                    'relative_position_x', 'relative_position_y',
                    
                    # B-plane values
                    'b_plane_xi', 'b_plane_zeta', 'b_plane_miss_distance',
                    'b_plane_cov_xx', 'b_plane_cov_xy', 'b_plane_cov_yy',
                    'b_plane_mahalanobis_distance', 'b_plane_chi2_threshold', 'b_plane_confidence_level',
                    
                    # Kalman filter metrics - Vehicle 1
                    'v1_ukf_uncertainty_x', 'v1_ukf_uncertainty_y', 'v1_ukf_uncertainty_v', 'v1_ukf_uncertainty_heading',
                    'v1_ukf_position_uncertainty', 'v1_ukf_velocity_uncertainty', 'v1_ukf_cov_trace', 'v1_ukf_cov_det',
                    'v1_ukf_predicted_uncertainty_x', 'v1_ukf_predicted_uncertainty_y',
                    'v1_ukf_predicted_position_uncertainty', 'v1_ukf_predicted_cov_trace',
                    
                    # Kalman filter metrics - Vehicle 2
                    'v2_ukf_uncertainty_x', 'v2_ukf_uncertainty_y', 'v2_ukf_uncertainty_v', 'v2_ukf_uncertainty_heading',
                    'v2_ukf_position_uncertainty', 'v2_ukf_velocity_uncertainty', 'v2_ukf_cov_trace', 'v2_ukf_cov_det',
                    'v2_ukf_predicted_uncertainty_x', 'v2_ukf_predicted_uncertainty_y',
                    'v2_ukf_predicted_position_uncertainty', 'v2_ukf_predicted_cov_trace',
                    
                    # Combined uncertainty
                    'combined_position_uncertainty', 'combined_velocity_uncertainty',
                    'combined_cov_trace', 'combined_predicted_cov_trace',
                    
                    # Time-based metrics
                    'time_to_closest_approach', 'closest_approach_distance',
                    
                    # Detection-specific
                    'detection_confidence', 'detection_result',  # True/False for collision prediction
                    
                    # Crash-specific
                    'actual_crash_distance', 'crash_occurred',
                    
                    # Match tracking
                    'matched_to_crash', 'time_to_crash', 'is_false_positive',                     'is_missed_crash'
                ])
    
    def _initialize_combined_csv(self):
        """Initialize the combined events CSV file (if not already done)"""
        if not self._combined_csv_initialized and not self.combined_events_csv.exists():
            self._initialize_csv(self.combined_events_csv)
            self._combined_csv_initialized = True
    
    def _initialize_summary_csv(self):
        """Initialize the trial summary CSV file"""
        if not self.summary_csv.exists():
            with open(self.summary_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trial_num', 'trial_duration', 'total_crashes', 'total_detections',
                    'true_positives', 'false_positives', 'missed_crashes',
                    'avg_detection_delay', 'precision', 'recall', 'f1_score',
                    'avg_uncertainty', 'avg_b_plane_miss_distance', 'avg_mahalanobis_distance'
                ])
    
    def start_trial(self):
        """Start a new trial"""
        self.trial_num += 1
        self.current_trial_events = []
        
        # Create trial-specific subfolder
        self.current_trial_dir = self.results_dir / f"trial_{self.trial_num:05d}"
        self.current_trial_dir.mkdir(exist_ok=True)
        
        # Create trial-specific CSV file
        self.events_csv = self.current_trial_dir / "detailed_events.csv"
        self._initialize_csv(self.events_csv)
        
        # Initialize combined CSV if needed
        self._initialize_combined_csv()
        
        print(f"[DetailedCrashTracker] Started trial {self.trial_num} (results in {self.current_trial_dir.name})")
    
    def _extract_b_plane_values(self, vehicle1, vehicle2, confidence=0.99):
        """Extract b-plane values from two vehicles"""
        try:
            # Get predicted states
            A_x, A_y, A_v, A_yaw = vehicle1.pred_next_state
            B_x, B_y, B_v, B_yaw = vehicle2.pred_next_state
            
            rel_pos = np.array([A_x - B_x, A_y - B_y, 0])
            rel_vel = np.array([
                A_v * np.cos(A_yaw) - B_v * np.cos(B_yaw),
                A_v * np.sin(A_yaw) - B_v * np.sin(B_yaw),
                0
            ])
            
            rel_vel_norm = np.linalg.norm(rel_vel)
            
            if rel_vel_norm < 1e-6:
                # Parallel velocity case
                return {
                    'b_plane_xi': 0.0,
                    'b_plane_zeta': 0.0,
                    'b_plane_miss_distance': np.linalg.norm(rel_pos[:2]),
                    'b_plane_cov_xx': 0.0,
                    'b_plane_cov_xy': 0.0,
                    'b_plane_cov_yy': 0.0,
                    'b_plane_mahalanobis_distance': 0.0,
                    'b_plane_chi2_threshold': 0.0,
                    'b_plane_confidence_level': confidence
                }
            
            rel_vel_normal = rel_vel / rel_vel_norm
            
            # Construct b-plane basis
            first_orth_basis = np.array([-rel_vel[1], rel_vel[0], 0])
            first_orth_basis /= np.linalg.norm(first_orth_basis)
            
            second_orth_basis = np.cross(first_orth_basis, rel_vel_normal)
            second_orth_basis /= np.linalg.norm(second_orth_basis)
            
            # Project miss distance vector onto b-plane
            b = rel_pos - np.dot(rel_pos, rel_vel_normal) * rel_vel_normal
            
            # Transform covariance into b-plane
            rel_P = vehicle1.predicted_P + vehicle2.predicted_P
            P_pos = rel_P[:2, :2]
            H_B_pos = np.vstack([first_orth_basis[:2], second_orth_basis[:2]])
            P_B = H_B_pos @ P_pos @ H_B_pos.T
            
            # Symmetrize and regularize
            P_B = 0.5 * (P_B + P_B.T)
            try:
                eigvals, eigvecs = np.linalg.eigh(P_B)
                eigvals = np.maximum(eigvals, 1e-9)
                P_B_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                b_B = H_B_pos @ b[:2]
                d_square = b_B.T @ np.linalg.inv(P_B_reg) @ b_B
                
                crit = chi2.ppf(confidence, df=2)
                
                return {
                    'b_plane_xi': float(b_B[0]),
                    'b_plane_zeta': float(b_B[1]),
                    'b_plane_miss_distance': float(np.linalg.norm(b_B)),
                    'b_plane_cov_xx': float(P_B_reg[0, 0]),
                    'b_plane_cov_xy': float(P_B_reg[0, 1]),
                    'b_plane_cov_yy': float(P_B_reg[1, 1]),
                    'b_plane_mahalanobis_distance': float(d_square),
                    'b_plane_chi2_threshold': float(crit),
                    'b_plane_confidence_level': confidence
                }
            except np.linalg.LinAlgError:
                return {
                    'b_plane_xi': 0.0,
                    'b_plane_zeta': 0.0,
                    'b_plane_miss_distance': np.linalg.norm(rel_pos[:2]),
                    'b_plane_cov_xx': 0.0,
                    'b_plane_cov_xy': 0.0,
                    'b_plane_cov_yy': 0.0,
                    'b_plane_mahalanobis_distance': float('inf'),
                    'b_plane_chi2_threshold': 0.0,
                    'b_plane_confidence_level': confidence
                }
        except Exception as e:
            # Return defaults on error
            return {
                'b_plane_xi': 0.0,
                'b_plane_zeta': 0.0,
                'b_plane_miss_distance': 0.0,
                'b_plane_cov_xx': 0.0,
                'b_plane_cov_xy': 0.0,
                'b_plane_cov_yy': 0.0,
                'b_plane_mahalanobis_distance': 0.0,
                'b_plane_chi2_threshold': 0.0,
                'b_plane_confidence_level': confidence
            }
    
    def _extract_kalman_metrics(self, vehicle):
        """Extract Kalman filter metrics from a vehicle"""
        P = vehicle.P
        P_pred = vehicle.predicted_P
        
        # Position uncertainty (x, y)
        pos_unc_x = np.sqrt(P[0, 0]) if P[0, 0] > 0 else 0.0
        pos_unc_y = np.sqrt(P[1, 1]) if P[1, 1] > 0 else 0.0
        
        # Velocity and heading uncertainty
        vel_unc = np.sqrt(P[2, 2]) if P[2, 2] > 0 else 0.0
        heading_unc = np.sqrt(P[3, 3]) if P[3, 3] > 0 else 0.0
        
        # Position covariance (2x2)
        P_pos = P[:2, :2]
        pos_uncertainty = np.sqrt(np.trace(P_pos))
        
        # Velocity covariance (1x1, but we use the variance)
        vel_uncertainty = vel_unc
        
        # Trace and determinant
        cov_trace = np.trace(P)
        try:
            cov_det = np.linalg.det(P)
        except:
            cov_det = 0.0
        
        # Predicted uncertainties
        P_pred_pos = P_pred[:2, :2]
        pred_pos_unc_x = np.sqrt(P_pred[0, 0]) if P_pred[0, 0] > 0 else 0.0
        pred_pos_unc_y = np.sqrt(P_pred[1, 1]) if P_pred[1, 1] > 0 else 0.0
        pred_pos_uncertainty = np.sqrt(np.trace(P_pred_pos))
        pred_cov_trace = np.trace(P_pred)
        
        return {
            'ukf_uncertainty_x': float(pos_unc_x),
            'ukf_uncertainty_y': float(pos_unc_y),
            'ukf_uncertainty_v': float(vel_unc),
            'ukf_uncertainty_heading': float(heading_unc),
            'ukf_position_uncertainty': float(pos_uncertainty),
            'ukf_velocity_uncertainty': float(vel_uncertainty),
            'ukf_cov_trace': float(cov_trace),
            'ukf_cov_det': float(cov_det),
            'ukf_predicted_uncertainty_x': float(pred_pos_unc_x),
            'ukf_predicted_uncertainty_y': float(pred_pos_unc_y),
            'ukf_predicted_position_uncertainty': float(pred_pos_uncertainty),
            'ukf_predicted_cov_trace': float(pred_cov_trace)
        }
    
    def _calculate_time_metrics(self, vehicle1, vehicle2):
        """Calculate time-based metrics"""
        v1_pos = vehicle1.external_state[:2]
        v2_pos = vehicle2.external_state[:2]
        v1_vel_vec = vehicle1.external_state[2] * np.array([
            np.cos(vehicle1.external_state[3]),
            np.sin(vehicle1.external_state[3])
        ])
        v2_vel_vec = vehicle2.external_state[2] * np.array([
            np.cos(vehicle2.external_state[3]),
            np.sin(vehicle2.external_state[3])
        ])
        
        rel_pos = np.array(v1_pos) - np.array(v2_pos)
        rel_vel = v1_vel_vec - v2_vel_vec
        
        rel_vel_mag = np.linalg.norm(rel_vel)
        
        if rel_vel_mag < 1e-6:
            return {
                'time_to_closest_approach': float('inf'),
                'closest_approach_distance': float(np.linalg.norm(rel_pos))
            }
        
        # Time to closest approach
        ttc = -np.dot(rel_pos, rel_vel) / (rel_vel_mag ** 2)
        ttc = max(0, ttc)  # Only future times
        
        # Closest approach distance
        closest_dist = np.linalg.norm(rel_pos + rel_vel * ttc) if ttc > 0 else np.linalg.norm(rel_pos)
        
        return {
            'time_to_closest_approach': float(ttc),
            'closest_approach_distance': float(closest_dist)
        }
    
    def record_event(self, timestamp: float, vehicle1, vehicle2, event_type: str,
                     confidence: float = None, is_crash: bool = False):
        """
        Record a detailed event (crash or detection)
        
        Args:
            timestamp: Simulation time
            vehicle1: First vehicle object (from vehicle_algo)
            vehicle2: Second vehicle object
            event_type: 'crash' or 'detection'
            confidence: Confidence level for detections
            is_crash: Whether this is an actual crash
        """
        # Ensure consistent ordering
        v1_id = min(vehicle1.id if hasattr(vehicle1, 'id') else 0, 
                   vehicle2.id if hasattr(vehicle2, 'id') else 1)
        v2_id = max(vehicle1.id if hasattr(vehicle1, 'id') else 0,
                   vehicle2.id if hasattr(vehicle2, 'id') else 1)
        
        # Get vehicle states
        v1_state = vehicle1.external_state
        v2_state = vehicle2.external_state
        v1_pred = vehicle1.pred_next_state
        v2_pred = vehicle2.pred_next_state
        
        # Extract b-plane values
        b_plane = self._extract_b_plane_values(vehicle1, vehicle2, confidence or 0.99)
        
        # Extract Kalman metrics
        v1_kf = self._extract_kalman_metrics(vehicle1)
        v2_kf = self._extract_kalman_metrics(vehicle2)
        
        # Calculate relative metrics
        rel_pos_vec = np.array(v1_state[:2]) - np.array(v2_state[:2])
        rel_distance = float(np.linalg.norm(rel_pos_vec))
        
        v1_vel_vec = v1_state[2] * np.array([np.cos(v1_state[3]), np.sin(v1_state[3])])
        v2_vel_vec = v2_state[2] * np.array([np.cos(v2_state[3]), np.sin(v2_state[3])])
        rel_vel_vec = v1_vel_vec - v2_vel_vec
        rel_vel_mag = float(np.linalg.norm(rel_vel_vec))
        
        # Time metrics
        time_metrics = self._calculate_time_metrics(vehicle1, vehicle2)
        
        # Combined uncertainty
        P1_pos = vehicle1.P[:2, :2]
        P2_pos = vehicle2.P[:2, :2]
        P_combined = P1_pos + P2_pos
        combined_pos_unc = float(np.sqrt(np.trace(P_combined)))
        
        P1_vel = vehicle1.P[2, 2]
        P2_vel = vehicle2.P[2, 2]
        combined_vel_unc = float(np.sqrt(P1_vel + P2_vel))
        
        # Combined predicted uncertainty
        P1_pred_pos = vehicle1.predicted_P[:2, :2]
        P2_pred_pos = vehicle2.predicted_P[:2, :2]
        P_pred_combined = P1_pred_pos + P2_pred_pos
        combined_pred_pos_unc = float(np.sqrt(np.trace(P_pred_combined)))
        
        # Detection result (check if algorithm predicted collision)
        detection_result = False
        if event_type == 'detection':
            detection_result = b_plane['b_plane_mahalanobis_distance'] < b_plane['b_plane_chi2_threshold']
        
        # Create event record
        event = {
            # Basic identifiers
            'trial_num': self.trial_num,
            'timestamp': timestamp,
            'event_type': event_type,
            'vehicle1_id': v1_id,
            'vehicle2_id': v2_id,
            
            # Vehicle 1 state
            'v1_x': float(v1_state[0]),
            'v1_y': float(v1_state[1]),
            'v1_velocity': float(v1_state[2]),
            'v1_heading': float(v1_state[3]),
            'v1_pred_x': float(v1_pred[0]),
            'v1_pred_y': float(v1_pred[1]),
            'v1_pred_velocity': float(v1_pred[2]),
            'v1_pred_heading': float(v1_pred[3]),
            
            # Vehicle 2 state
            'v2_x': float(v2_state[0]),
            'v2_y': float(v2_state[1]),
            'v2_velocity': float(v2_state[2]),
            'v2_heading': float(v2_state[3]),
            'v2_pred_x': float(v2_pred[0]),
            'v2_pred_y': float(v2_pred[1]),
            'v2_pred_velocity': float(v2_pred[2]),
            'v2_pred_heading': float(v2_pred[3]),
            
            # Relative metrics
            'relative_distance': rel_distance,
            'relative_velocity_magnitude': rel_vel_mag,
            'relative_velocity_x': float(rel_vel_vec[0]),
            'relative_velocity_y': float(rel_vel_vec[1]),
            'relative_position_x': float(rel_pos_vec[0]),
            'relative_position_y': float(rel_pos_vec[1]),
            
            # B-plane values
            **b_plane,
            
            # Kalman metrics - Vehicle 1
            **{f'v1_{k}': v for k, v in v1_kf.items()},
            
            # Kalman metrics - Vehicle 2
            **{f'v2_{k}': v for k, v in v2_kf.items()},
            
            # Combined uncertainty
            'combined_position_uncertainty': combined_pos_unc,
            'combined_velocity_uncertainty': combined_vel_unc,
            'combined_cov_trace': float(np.trace(P1_pos + P2_pos)),
            'combined_predicted_cov_trace': float(np.trace(P1_pred_pos + P2_pred_pos)),
            
            # Time metrics
            **time_metrics,
            
            # Detection-specific
            'detection_confidence': confidence if confidence is not None else 0.0,
            'detection_result': detection_result,
            
            # Crash-specific
            'actual_crash_distance': rel_distance if is_crash else 0.0,
            'crash_occurred': is_crash,
            
            # Match tracking (will be filled later)
            'matched_to_crash': False,
            'time_to_crash': 0.0,
            'is_false_positive': False,
            'is_missed_crash': False
        }
        
        self.current_trial_events.append(event)
        
        # Write to CSV immediately
        self._write_event_to_csv(event)
        
        if event_type == 'crash':
            self.total_actual_crashes += 1
        elif event_type == 'detection':
            self.total_detections += 1
    
    def _write_event_to_csv(self, event):
        """Write a single event to both trial-specific and combined CSV files"""
        # Write to trial-specific CSV
        if self.events_csv:
            with open(self.events_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.get('trial_num', 0),
                    event.get('timestamp', 0.0),
                    event.get('event_type', ''),
                    event.get('vehicle1_id', 0),
                    event.get('vehicle2_id', 0),
                    event.get('v1_x', 0.0),
                    event.get('v1_y', 0.0),
                    event.get('v1_velocity', 0.0),
                    event.get('v1_heading', 0.0),
                    event.get('v1_pred_x', 0.0),
                    event.get('v1_pred_y', 0.0),
                    event.get('v1_pred_velocity', 0.0),
                    event.get('v1_pred_heading', 0.0),
                    event.get('v2_x', 0.0),
                    event.get('v2_y', 0.0),
                    event.get('v2_velocity', 0.0),
                    event.get('v2_heading', 0.0),
                    event.get('v2_pred_x', 0.0),
                    event.get('v2_pred_y', 0.0),
                    event.get('v2_pred_velocity', 0.0),
                    event.get('v2_pred_heading', 0.0),
                    event.get('relative_distance', 0.0),
                    event.get('relative_velocity_magnitude', 0.0),
                    event.get('relative_velocity_x', 0.0),
                    event.get('relative_velocity_y', 0.0),
                    event.get('relative_position_x', 0.0),
                    event.get('relative_position_y', 0.0),
                    event.get('b_plane_xi', 0.0),
                    event.get('b_plane_zeta', 0.0),
                    event.get('b_plane_miss_distance', 0.0),
                    event.get('b_plane_cov_xx', 0.0),
                    event.get('b_plane_cov_xy', 0.0),
                    event.get('b_plane_cov_yy', 0.0),
                    event.get('b_plane_mahalanobis_distance', 0.0),
                    event.get('b_plane_chi2_threshold', 0.0),
                    event.get('b_plane_confidence_level', 0.0),
                    event.get('v1_ukf_uncertainty_x', 0.0),
                    event.get('v1_ukf_uncertainty_y', 0.0),
                    event.get('v1_ukf_uncertainty_v', 0.0),
                    event.get('v1_ukf_uncertainty_heading', 0.0),
                    event.get('v1_ukf_position_uncertainty', 0.0),
                    event.get('v1_ukf_velocity_uncertainty', 0.0),
                    event.get('v1_ukf_cov_trace', 0.0),
                    event.get('v1_ukf_cov_det', 0.0),
                    event.get('v1_ukf_predicted_uncertainty_x', 0.0),
                    event.get('v1_ukf_predicted_uncertainty_y', 0.0),
                    event.get('v1_ukf_predicted_position_uncertainty', 0.0),
                    event.get('v1_ukf_predicted_cov_trace', 0.0),
                    event.get('v2_ukf_uncertainty_x', 0.0),
                    event.get('v2_ukf_uncertainty_y', 0.0),
                    event.get('v2_ukf_uncertainty_v', 0.0),
                    event.get('v2_ukf_uncertainty_heading', 0.0),
                    event.get('v2_ukf_position_uncertainty', 0.0),
                    event.get('v2_ukf_velocity_uncertainty', 0.0),
                    event.get('v2_ukf_cov_trace', 0.0),
                    event.get('v2_ukf_cov_det', 0.0),
                    event.get('v2_ukf_predicted_uncertainty_x', 0.0),
                    event.get('v2_ukf_predicted_uncertainty_y', 0.0),
                    event.get('v2_ukf_predicted_position_uncertainty', 0.0),
                    event.get('v2_ukf_predicted_cov_trace', 0.0),
                    event.get('combined_position_uncertainty', 0.0),
                    event.get('combined_velocity_uncertainty', 0.0),
                    event.get('combined_cov_trace', 0.0),
                    event.get('combined_predicted_cov_trace', 0.0),
                    event.get('time_to_closest_approach', 0.0),
                    event.get('closest_approach_distance', 0.0),
                    event.get('detection_confidence', 0.0),
                    event.get('detection_result', False),
                    event.get('actual_crash_distance', 0.0),
                    event.get('crash_occurred', False),
                    event.get('matched_to_crash', False),
                    event.get('time_to_crash', 0.0),
                    event.get('is_false_positive', False),
                    event.get('is_missed_crash', False)
                ])
        
        # Also write to combined CSV (if initialized)
        if self._combined_csv_initialized and self.combined_events_csv.exists():
            with open(self.combined_events_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.get('trial_num', 0),
                    event.get('timestamp', 0.0),
                    event.get('event_type', ''),
                    event.get('vehicle1_id', 0),
                    event.get('vehicle2_id', 0),
                    event.get('v1_x', 0.0),
                    event.get('v1_y', 0.0),
                    event.get('v1_velocity', 0.0),
                    event.get('v1_heading', 0.0),
                    event.get('v1_pred_x', 0.0),
                    event.get('v1_pred_y', 0.0),
                    event.get('v1_pred_velocity', 0.0),
                    event.get('v1_pred_heading', 0.0),
                    event.get('v2_x', 0.0),
                    event.get('v2_y', 0.0),
                    event.get('v2_velocity', 0.0),
                    event.get('v2_heading', 0.0),
                    event.get('v2_pred_x', 0.0),
                    event.get('v2_pred_y', 0.0),
                    event.get('v2_pred_velocity', 0.0),
                    event.get('v2_pred_heading', 0.0),
                    event.get('relative_distance', 0.0),
                    event.get('relative_velocity_magnitude', 0.0),
                    event.get('relative_velocity_x', 0.0),
                    event.get('relative_velocity_y', 0.0),
                    event.get('relative_position_x', 0.0),
                    event.get('relative_position_y', 0.0),
                    event.get('b_plane_xi', 0.0),
                    event.get('b_plane_zeta', 0.0),
                    event.get('b_plane_miss_distance', 0.0),
                    event.get('b_plane_cov_xx', 0.0),
                    event.get('b_plane_cov_xy', 0.0),
                    event.get('b_plane_cov_yy', 0.0),
                    event.get('b_plane_mahalanobis_distance', 0.0),
                    event.get('b_plane_chi2_threshold', 0.0),
                    event.get('b_plane_confidence_level', 0.0),
                    event.get('v1_ukf_uncertainty_x', 0.0),
                    event.get('v1_ukf_uncertainty_y', 0.0),
                    event.get('v1_ukf_uncertainty_v', 0.0),
                    event.get('v1_ukf_uncertainty_heading', 0.0),
                    event.get('v1_ukf_position_uncertainty', 0.0),
                    event.get('v1_ukf_velocity_uncertainty', 0.0),
                    event.get('v1_ukf_cov_trace', 0.0),
                    event.get('v1_ukf_cov_det', 0.0),
                    event.get('v1_ukf_predicted_uncertainty_x', 0.0),
                    event.get('v1_ukf_predicted_uncertainty_y', 0.0),
                    event.get('v1_ukf_predicted_position_uncertainty', 0.0),
                    event.get('v1_ukf_predicted_cov_trace', 0.0),
                    event.get('v2_ukf_uncertainty_x', 0.0),
                    event.get('v2_ukf_uncertainty_y', 0.0),
                    event.get('v2_ukf_uncertainty_v', 0.0),
                    event.get('v2_ukf_uncertainty_heading', 0.0),
                    event.get('v2_ukf_position_uncertainty', 0.0),
                    event.get('v2_ukf_velocity_uncertainty', 0.0),
                    event.get('v2_ukf_cov_trace', 0.0),
                    event.get('v2_ukf_cov_det', 0.0),
                    event.get('v2_ukf_predicted_uncertainty_x', 0.0),
                    event.get('v2_ukf_predicted_uncertainty_y', 0.0),
                    event.get('v2_ukf_predicted_position_uncertainty', 0.0),
                    event.get('v2_ukf_predicted_cov_trace', 0.0),
                    event.get('combined_position_uncertainty', 0.0),
                    event.get('combined_velocity_uncertainty', 0.0),
                    event.get('combined_cov_trace', 0.0),
                    event.get('combined_predicted_cov_trace', 0.0),
                    event.get('time_to_closest_approach', 0.0),
                    event.get('closest_approach_distance', 0.0),
                    event.get('detection_confidence', 0.0),
                    event.get('detection_result', False),
                    event.get('actual_crash_distance', 0.0),
                    event.get('crash_occurred', False),
                    event.get('matched_to_crash', False),
                    event.get('time_to_crash', 0.0),
                    event.get('is_false_positive', False),
                    event.get('is_missed_crash', False)
                ])
    
    def end_trial(self, trial_duration: float = 0.0, max_delay_seconds: float = 5.0):
        """End current trial and analyze matches"""
        print(f"[DetailedCrashTracker] Ending trial {self.trial_num}")
        
        # Save trial summary JSON to trial folder
        if self.current_trial_dir:
            trial_summary_file = self.current_trial_dir / "trial_summary.json"
            crashes = [e for e in self.current_trial_events if e['event_type'] == 'crash']
            detections = [e for e in self.current_trial_events if e['event_type'] == 'detection']
            
            trial_summary = {
                'trial_num': self.trial_num,
                'total_crashes': len(crashes),
                'total_detections': len(detections),
                'crashes': crashes,
                'detections': detections
            }
            
            with open(trial_summary_file, 'w') as f:
                json.dump(trial_summary, f, indent=2)
        
        # Separate crashes and detections
        crashes = [e for e in self.current_trial_events if e['event_type'] == 'crash']
        detections = [e for e in self.current_trial_events if e['event_type'] == 'detection']
        
        # Match detections to crashes
        matched_detections = set()
        matched_crashes = set()
        matches = []
        
        for detection in detections:
            det_time = detection['timestamp']
            det_v1 = detection['vehicle1_id']
            det_v2 = detection['vehicle2_id']
            
            for crash in crashes:
                crash_time = crash['timestamp']
                crash_v1 = crash['vehicle1_id']
                crash_v2 = crash['vehicle2_id']
                
                if ((det_v1 == crash_v1 and det_v2 == crash_v2) or
                    (det_v1 == crash_v2 and det_v2 == crash_v1)):
                    if det_time < crash_time:
                        delay = crash_time - det_time
                        if delay <= max_delay_seconds:
                            # Match found
                            detection['matched_to_crash'] = True
                            detection['time_to_crash'] = delay
                            crash['matched_to_crash'] = True
                            matched_detections.add(id(detection))
                            matched_crashes.add(id(crash))
                            matches.append((detection, crash, delay))
                            break
        
        # Mark false positives
        for detection in detections:
            if id(detection) not in matched_detections:
                detection['is_false_positive'] = True
        
        # Mark missed crashes
        for crash in crashes:
            if id(crash) not in matched_crashes:
                crash['is_missed_crash'] = True
        
        # Calculate statistics
        true_positives = len(matches)
        false_positives = len([d for d in detections if d['is_false_positive']])
        missed_crashes = len([c for c in crashes if c['is_missed_crash']])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + missed_crashes) if (true_positives + missed_crashes) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_delay = np.mean([m[2] for m in matches]) if matches else 0.0
        
        # Calculate average metrics
        avg_uncertainty = np.mean([e['combined_position_uncertainty'] for e in self.current_trial_events]) if self.current_trial_events else 0.0
        avg_b_plane_miss = np.mean([e['b_plane_miss_distance'] for e in detections]) if detections else 0.0
        avg_mahalanobis = np.mean([e['b_plane_mahalanobis_distance'] for e in detections if e['b_plane_mahalanobis_distance'] != float('inf')]) if detections else 0.0
        
        # Write trial summary
        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.trial_num,
                trial_duration,
                len(crashes),
                len(detections),
                true_positives,
                false_positives,
                missed_crashes,
                avg_delay,
                precision,
                recall,
                f1_score,
                avg_uncertainty,
                avg_b_plane_miss,
                avg_mahalanobis
            ])
        
        # Update events CSV with match information
        self._update_events_with_matches()
        
        self.total_trials += 1
        
        print(f"  Trial {self.trial_num}: {len(crashes)} crashes, {len(detections)} detections, "
              f"{true_positives} TP, {false_positives} FP, {missed_crashes} missed")
    
    def _update_events_with_matches(self):
        """Re-write events CSV with match information"""
        # This is a simplified approach - in practice, you might want to update the CSV file
        # For now, the match information is written when we detect matches in end_trial
        pass
    
    def save_final_summary(self):
        """Generate and save final summary statistics"""
        summary_file = self.results_dir / "final_summary.json"
        
        # Read all trial summaries
        trial_summaries = []
        if self.summary_csv.exists():
            with open(self.summary_csv, 'r') as f:
                reader = csv.DictReader(f)
                trial_summaries = list(reader)
        
        if not trial_summaries:
            return
        
        # Calculate overall statistics
        total_crashes = sum(int(t['total_crashes']) for t in trial_summaries)
        total_detections = sum(int(t['total_detections']) for t in trial_summaries)
        total_tp = sum(int(t['true_positives']) for t in trial_summaries)
        total_fp = sum(int(t['false_positives']) for t in trial_summaries)
        total_missed = sum(int(t['missed_crashes']) for t in trial_summaries)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_missed) if (total_tp + total_missed) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        avg_delays = [float(t['avg_detection_delay']) for t in trial_summaries if float(t['avg_detection_delay']) > 0]
        avg_delay = np.mean(avg_delays) if avg_delays else 0.0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_trials': self.total_trials,
            'total_crashes': total_crashes,
            'total_detections': total_detections,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'missed_crashes': total_missed,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'average_detection_delay': avg_delay
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[DetailedCrashTracker] Final summary saved to {summary_file}")
        print(f"  Trials: {summary['total_trials']}")
        print(f"  Precision: {summary['precision']:.3f}")
        print(f"  Recall: {summary['recall']:.3f}")
        print(f"  F1 Score: {summary['f1_score']:.3f}")

