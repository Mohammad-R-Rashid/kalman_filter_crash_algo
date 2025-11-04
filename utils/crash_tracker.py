"""
Crash Tracker - Tracks actual crashes and crash detections
Compares detection predictions with actual physical collisions
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class CrashTracker:
    """Tracks actual crashes and crash detections over simulation runs"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Per-trial tracking
        self.trial_num = 0
        self.current_trial_crashes = []  # List of (timestamp, vehicle1_id, vehicle2_id)
        self.current_trial_detections = []  # List of (timestamp, vehicle1_id, vehicle2_id, confidence)
        
        # Global statistics across all trials
        self.total_trials = 0
        self.total_actual_crashes = 0
        self.total_detections = 0
        
        # Match tracking (detection -> actual crash)
        self.detection_to_crash_matches = []  # Detections that matched a crash
        self.detection_to_crash_delays = []  # Time delays between detection and crash
        self.missed_crashes = []  # Crashes that had no prior detection
        self.false_positives = []  # Detections that never resulted in a crash
        
        # Real-time results file
        self.realtime_file = self.results_dir / "realtime_results.json"
        
        # Initialize real-time results
        self._initialize_realtime_file()
    
    def _initialize_realtime_file(self):
        """Initialize the real-time results JSON file"""
        if not self.realtime_file.exists():
            initial_data = {
                "trials_completed": 0,
                "total_actual_crashes": 0,
                "total_detections": 0,
                "detection_matches": 0,
                "missed_crashes": 0,
                "false_positives": 0,
                "average_detection_delay": 0.0,
                "latest_trial_stats": {}
            }
            with open(self.realtime_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def start_trial(self):
        """Start a new trial"""
        self.trial_num += 1
        self.current_trial_crashes = []
        self.current_trial_detections = []
        print(f"[CrashTracker] Started trial {self.trial_num}")
    
    def record_actual_crash(self, timestamp: float, vehicle1_id: int, vehicle2_id: int):
        """Record an actual physical crash (vehicles touching)"""
        # Ensure unique pair (smaller ID first)
        v1, v2 = min(vehicle1_id, vehicle2_id), max(vehicle1_id, vehicle2_id)
        
        crash_entry = {
            "timestamp": timestamp,
            "vehicle1_id": v1,
            "vehicle2_id": v2,
            "trial": self.trial_num
        }
        
        # Check if this crash was already recorded (avoid duplicates)
        if not any(c["timestamp"] == timestamp and 
                  c["vehicle1_id"] == v1 and 
                  c["vehicle2_id"] == v2 
                  for c in self.current_trial_crashes):
            self.current_trial_crashes.append(crash_entry)
            self.total_actual_crashes += 1
            print(f"[CrashTracker] Actual crash recorded at t={timestamp:.2f}s: V{v1} <-> V{v2}")
    
    def record_detection(self, timestamp: float, vehicle1_id: int, vehicle2_id: int, confidence: float = 0.95):
        """Record a crash detection (algorithm prediction)"""
        # Ensure unique pair (smaller ID first)
        v1, v2 = min(vehicle1_id, vehicle2_id), max(vehicle1_id, vehicle2_id)
        
        detection_entry = {
            "timestamp": timestamp,
            "vehicle1_id": v1,
            "vehicle2_id": v2,
            "confidence": confidence,
            "trial": self.trial_num
        }
        
        # Check if this detection was already recorded (avoid duplicates in same timestep)
        if not any(d["timestamp"] == timestamp and 
                  d["vehicle1_id"] == v1 and 
                  d["vehicle2_id"] == v2 
                  for d in self.current_trial_detections):
            self.current_trial_detections.append(detection_entry)
            self.total_detections += 1
    
    def end_trial(self, max_delay_seconds: float = 5.0):
        """End current trial and analyze matches between detections and crashes"""
        print(f"[CrashTracker] Ending trial {self.trial_num}")
        
        # Match detections to crashes
        # A detection matches if it occurs before a crash for the same vehicle pair
        # within max_delay_seconds
        
        matched_detections = set()
        matched_crashes = set()
        
        for detection in self.current_trial_detections:
            det_time = detection["timestamp"]
            det_v1 = detection["vehicle1_id"]
            det_v2 = detection["vehicle2_id"]
            
            # Find matching crash (same vehicle pair, within time window)
            for crash in self.current_trial_crashes:
                crash_time = crash["timestamp"]
                crash_v1 = crash["vehicle1_id"]
                crash_v2 = crash["vehicle2_id"]
                
                # Check if same vehicle pair
                if (det_v1 == crash_v1 and det_v2 == crash_v2) or \
                   (det_v1 == crash_v2 and det_v2 == crash_v1):
                    # Check if detection occurred before crash
                    if det_time < crash_time:
                        # Check if within time window
                        delay = crash_time - det_time
                        if delay <= max_delay_seconds:
                            # Match found!
                            self.detection_to_crash_matches.append({
                                "detection_time": det_time,
                                "crash_time": crash_time,
                                "delay": delay,
                                "vehicle1_id": det_v1,
                                "vehicle2_id": det_v2,
                                "trial": self.trial_num
                            })
                            self.detection_to_crash_delays.append(delay)
                            matched_detections.add(id(detection))
                            matched_crashes.add(id(crash))
                            break
        
        # Find missed crashes (crashes with no prior detection)
        for crash in self.current_trial_crashes:
            if id(crash) not in matched_crashes:
                self.missed_crashes.append({
                    "timestamp": crash["timestamp"],
                    "vehicle1_id": crash["vehicle1_id"],
                    "vehicle2_id": crash["vehicle2_id"],
                    "trial": self.trial_num
                })
        
        # Find false positives (detections with no subsequent crash)
        for detection in self.current_trial_detections:
            if id(detection) not in matched_detections:
                self.false_positives.append({
                    "timestamp": detection["timestamp"],
                    "vehicle1_id": detection["vehicle1_id"],
                    "vehicle2_id": detection["vehicle2_id"],
                    "confidence": detection["confidence"],
                    "trial": self.trial_num
                })
        
        self.total_trials += 1
        
        # Update real-time results
        self._update_realtime_results()
        
        # Save trial results
        self._save_trial_results()
    
    def _update_realtime_results(self):
        """Update real-time results JSON file"""
        avg_delay = np.mean(self.detection_to_crash_delays) if self.detection_to_crash_delays else 0.0
        
        data = {
            "trials_completed": self.total_trials,
            "total_actual_crashes": self.total_actual_crashes,
            "total_detections": self.total_detections,
            "detection_matches": len(self.detection_to_crash_matches),
            "missed_crashes": len(self.missed_crashes),
            "false_positives": len(self.false_positives),
            "average_detection_delay": float(avg_delay),
            "latest_trial_stats": {
                "trial_num": self.trial_num,
                "crashes": len(self.current_trial_crashes),
                "detections": len(self.current_trial_detections),
                "matches": len([m for m in self.detection_to_crash_matches if m["trial"] == self.trial_num]),
                "missed": len([m for m in self.missed_crashes if m["trial"] == self.trial_num]),
                "false_positives": len([f for f in self.false_positives if f["trial"] == self.trial_num])
            }
        }
        
        with open(self.realtime_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_trial_results(self):
        """Save detailed results for the current trial"""
        trial_file = self.results_dir / f"trial_{self.trial_num:05d}.json"
        
        trial_data = {
            "trial_num": self.trial_num,
            "crashes": self.current_trial_crashes,
            "detections": self.current_trial_detections,
            "matches": [m for m in self.detection_to_crash_matches if m["trial"] == self.trial_num],
            "missed_crashes": [m for m in self.missed_crashes if m["trial"] == self.trial_num],
            "false_positives": [f for f in self.false_positives if f["trial"] == self.trial_num]
        }
        
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2)
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics across all trials"""
        if self.total_trials == 0:
            return {}
        
        avg_delay = np.mean(self.detection_to_crash_delays) if self.detection_to_crash_delays else 0.0
        std_delay = np.std(self.detection_to_crash_delays) if self.detection_to_crash_delays else 0.0
        
        # Calculate metrics
        true_positives = len(self.detection_to_crash_matches)
        false_positives_count = len(self.false_positives)
        false_negatives = len(self.missed_crashes)
        
        precision = true_positives / (true_positives + false_positives_count) if (true_positives + false_positives_count) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        summary = {
            "total_trials": self.total_trials,
            "total_actual_crashes": self.total_actual_crashes,
            "total_detections": self.total_detections,
            "true_positives": true_positives,
            "false_positives": false_positives_count,
            "false_negatives": false_negatives,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "average_detection_delay_seconds": float(avg_delay),
            "std_detection_delay_seconds": float(std_delay),
            "detection_matches": len(self.detection_to_crash_matches),
            "missed_crashes": len(self.missed_crashes),
            "false_positives_count": len(self.false_positives)
        }
        
        return summary
    
    def save_final_summary(self):
        """Save final summary to results directory"""
        summary = self.generate_summary()
        summary_file = self.results_dir / "final_summary.json"
        
        summary["timestamp"] = datetime.now().isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[CrashTracker] Final summary saved to {summary_file}")
        print(f"  Trials: {summary['total_trials']}")
        print(f"  Actual crashes: {summary['total_actual_crashes']}")
        print(f"  Detections: {summary['total_detections']}")
        print(f"  True positives: {summary['true_positives']}")
        print(f"  False positives: {summary['false_positives']}")
        print(f"  Missed crashes: {summary['false_negatives']}")
        print(f"  Precision: {summary['precision']:.3f}")
        print(f"  Recall: {summary['recall']:.3f}")
        print(f"  F1 Score: {summary['f1_score']:.3f}")
        print(f"  Avg detection delay: {summary['average_detection_delay_seconds']:.3f}s")

