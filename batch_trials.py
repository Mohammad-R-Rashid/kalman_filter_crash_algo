"""
Batch Trials Runner - Run multiple simulation trials without UI for crash tracking
"""
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
from utils.vehicle_algo import vehicle as VehicleAlgo, vehicle_simulation
from utils.crash_tracker import CrashTracker
from utils.detailed_crash_tracker import DetailedCrashTracker

# Simulation parameters (matching game.py)
NUM_AGENTS = 60
SPEED_MIN = 3.0
SPEED_MAX = 20.0
VEHICLE_LENGTH = 2.5
VEHICLE_WIDTH = 2.0
MAX_STEERING_ANGLE = math.radians(30)
SIM_DT = 0.1
PREDICTION_UPDATE_INTERVAL = 0.2
LANE_WIDTH = 3.5

# Trial parameters
MAX_TRIAL_TIME = 60.0  # seconds per trial
COLLISION_DISTANCE = 2.0  # meters for physical collision detection


class SimpleVehicle:
    """Simplified vehicle for batch processing (no rendering)"""
    
    def __init__(self, vehicle_id, initial_state, control_input):
        self.id = vehicle_id
        self.algo_vehicle = VehicleAlgo(
            L=VEHICLE_LENGTH,
            external_state=initial_state,
            internal_state=initial_state,
            control_input=control_input
        )
        self.target_speed = initial_state[2]
        self.sim_time = 0.0
    
    def update(self, dt):
        """Update vehicle state"""
        self.sim_time += dt
        
        # Simple control: maintain target speed
        x, y, v, psi = self.algo_vehicle.external_state
        accel = (self.target_speed - v) * 0.5
        accel = np.clip(accel, -3.0, 2.0)
        steering = 0.0  # Simplified: straight line movement for batch processing
        
        self.algo_vehicle.update_external([accel, steering], dt)


def create_random_vehicles(num_vehicles):
    """Create vehicles with random initial positions and speeds"""
    vehicles = []
    
    for i in range(num_vehicles):
        # Random initial position (within a reasonable range)
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        
        # Random speed
        speed_class = random.random()
        if speed_class < 0.3:
            speed = random.uniform(SPEED_MIN, SPEED_MIN + 3.0)
        elif speed_class < 0.7:
            speed = random.uniform(SPEED_MIN + 3.0, SPEED_MAX - 3.0)
        else:
            speed = random.uniform(SPEED_MAX - 3.0, SPEED_MAX)
        
        # Random heading
        heading = random.uniform(0, 2 * math.pi)
        
        initial_state = [x, y, speed, heading]
        control_input = [0.0, 0.0]
        
        vehicles.append(SimpleVehicle(i, initial_state, control_input))
    
    return vehicles


def check_collisions(vehicles, crash_tracker, detailed_tracker, sim_time):
    """Check for collision detections (algorithm predictions)"""
    algo_vehicles = [v.algo_vehicle for v in vehicles]
    
    # Make sure all vehicles have predictions
    for v in vehicles:
        v.algo_vehicle.ukf_predict(SIM_DT)
    
    sim = vehicle_simulation(algo_vehicles)
    is_collision, idx1, idx2 = sim.is_collision(0.99)
    
    if crash_tracker is not None and is_collision and idx1 is not None and idx2 is not None:
        crash_tracker.record_detection(sim_time, idx1, idx2, 0.99)
    
    if detailed_tracker is not None and is_collision and idx1 is not None and idx2 is not None:
        vehicles[idx1].algo_vehicle.id = idx1
        vehicles[idx2].algo_vehicle.id = idx2
        detailed_tracker.record_event(sim_time, vehicles[idx1].algo_vehicle, vehicles[idx2].algo_vehicle,
                                     'detection', confidence=0.99, is_crash=False)
    
    return is_collision, idx1, idx2


def check_physical_crashes(vehicles, crash_tracker, detailed_tracker, sim_time):
    """Check for actual physical crashes"""
    algo_vehicles = [v.algo_vehicle for v in vehicles]
    sim = vehicle_simulation(algo_vehicles)
    
    is_crash, idx1, idx2 = sim.check_physical_collision(collision_distance=COLLISION_DISTANCE)
    
    if crash_tracker is not None and is_crash and idx1 is not None and idx2 is not None:
        crash_tracker.record_actual_crash(sim_time, idx1, idx2)
    
    if detailed_tracker is not None and is_crash and idx1 is not None and idx2 is not None:
        vehicles[idx1].algo_vehicle.id = idx1
        vehicles[idx2].algo_vehicle.id = idx2
        detailed_tracker.record_event(sim_time, vehicles[idx1].algo_vehicle, vehicles[idx2].algo_vehicle,
                                     'crash', confidence=None, is_crash=True)
    
    return is_crash, idx1, idx2


def run_batch_trials(num_trials=10):
    """Run multiple trials without UI"""
    print(f"Starting batch processing: {num_trials} trials")
    
    # Initialize crash trackers
    crash_tracker = CrashTracker(results_dir="results")
    detailed_tracker = DetailedCrashTracker(results_dir="results")
    
    for trial_num in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1}/{num_trials}")
        print(f"{'='*60}")
        
        crash_tracker.start_trial()
        detailed_tracker.start_trial()
        
        # Create vehicles
        vehicles = create_random_vehicles(NUM_AGENTS)
        
        # Simulation loop
        sim_time = 0.0
        collision_timer = 0.0
        step_count = 0
        
        while sim_time < MAX_TRIAL_TIME:
            # Update all vehicles
            for v in vehicles:
                v.update(SIM_DT)
            
            # Check for physical crashes
            physical_crash, _, _ = check_physical_crashes(vehicles, crash_tracker, detailed_tracker, sim_time)
            
            # Check for collision detections periodically
            collision_timer += SIM_DT
            if collision_timer >= PREDICTION_UPDATE_INTERVAL:
                collision_timer = 0.0
                check_collisions(vehicles, crash_tracker, detailed_tracker, sim_time)
            
            sim_time += SIM_DT
            step_count += 1
            
            # Progress indicator
            if step_count % 100 == 0:
                progress = (sim_time / MAX_TRIAL_TIME) * 100
                print(f"  Progress: {progress:.1f}% (t={sim_time:.1f}s)", end='\r')
        
        # End trial
        crash_tracker.end_trial()
        detailed_tracker.end_trial(trial_duration=sim_time)
        
        # Print trial summary
        trial_crashes = len([c for c in crash_tracker.current_trial_crashes])
        trial_detections = len([d for d in crash_tracker.current_trial_detections])
        print(f"\n  Trial {trial_num + 1} complete: {trial_crashes} crashes, {trial_detections} detections")
    
    # Save final summaries
    crash_tracker.save_final_summary()
    detailed_tracker.save_final_summary()
    
    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    num_trials = 10
    if len(sys.argv) > 1:
        try:
            num_trials = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of trials, using default: 10")
    
    run_batch_trials(num_trials)

