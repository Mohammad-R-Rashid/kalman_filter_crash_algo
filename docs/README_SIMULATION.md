# Vehicle Collision Prediction Simulation

A simulation environment for testing collision prediction algorithms based on celestial mechanics principles (b-plane approach), bicycle vehicle models, and Unscented Kalman Filter (UKF) state estimation.

## Overview

This simulation implements the concepts from `plan.md`:
- **Bicycle kinematic model** for vehicle dynamics
- **State vector**: `[x, y, v, ψ, ψ̇]` (position, velocity, heading, yaw rate)
- **State covariance tracking** (5×5 matrix) for uncertainty quantification
- **Trajectory prediction** at discrete time steps (0.1s intervals, 5s horizon)
- **Uncertainty ellipses** visualization showing 2-sigma confidence regions
- **Collision probability computation** using placeholder API (ready for integration)

## Features Implemented

### ✅ Vehicle State Estimation
- Full 5-dimensional state vector per vehicle
- State covariance matrix propagation (simplified UKF placeholder)
- Real-time state updates using bicycle kinematic model
- **Vehicles are constrained to roads** - they follow the road network graph with 70% road-constraint blending

### ✅ Prediction System
- Predicts trajectories 5 seconds into the future
- Propagates uncertainty forward in time
- Updates every 0.2 seconds for performance

### ✅ Collision Detection
- Pairwise collision probability computation
- Uses combined covariance (P_Δ = P_ego + P_other)
- Placeholder for b-plane projection API call
- Visual warnings for high-risk scenarios

### ✅ Visualization
- Uncertainty ellipses around current and predicted positions
- Predicted trajectory dots (fading over time)
- Color-coded vehicles (red = high collision risk)
- Real-time collision warnings on screen

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install pygame numpy
```

## Running the Simulation

```bash
python game.py
```

**Note**: Requires `wampus.json` in the same directory (your road network data).

## Building Visualization (Optional)

To add building footprints for better visual context:

```bash
python fetch_buildings.py
```

This will download building data from OpenStreetMap and save it to `buildings.json`. Buildings will be rendered as dark gray polygons beneath the roads, making it much clearer which areas are roads vs. structures.

If you skip this step, the simulation will still work fine - it just won't show buildings.

## Controls

- **Right-click + Drag**: Pan the view
- **Mouse Wheel**: Zoom in/out
- **R**: Reset view
- **U**: Toggle uncertainty ellipses on/off
- **P**: Toggle prediction trajectories on/off
- **B**: Toggle building visualization on/off
- **ESC/Close**: Exit

## Configuration Parameters

Edit these in `game.py` to customize:

```python
# Vehicle parameters
NUM_AGENTS = 20              # Number of vehicles
SPEED_MIN = 4.0             # Min speed (m/s)
SPEED_MAX = 10.0            # Max speed (m/s)
VEHICLE_LENGTH = 4.5        # Wheelbase (m)
MAX_STEERING_ANGLE = 30°    # Max steering

# Prediction parameters
DT = 0.1                    # Prediction timestep (s)
PREDICTION_HORIZON = 5.0    # How far to predict (s)
PREDICTION_UPDATE_INTERVAL = 0.2  # How often to recompute

# Process noise (UKF tuning)
PROCESS_NOISE_POS = 0.5     # Position uncertainty (m)
PROCESS_NOISE_VEL = 0.2     # Velocity uncertainty (m/s)
PROCESS_NOISE_HEADING = 0.05  # Heading uncertainty (rad)
PROCESS_NOISE_YAW_RATE = 0.02  # Yaw rate uncertainty (rad/s)

# Visualization
SHOW_UNCERTAINTY = True
SHOW_PREDICTIONS = True
ELLIPSE_SIGMA = 2.0         # Draw 2-sigma ellipses
COLLISION_THRESHOLD = 0.01  # Warning threshold
```

## Architecture

### Key Classes

**`VehicleState`**: State vector dataclass
```python
- x, y: position (meters)
- v: speed (m/s)
- psi: heading (radians)
- psi_dot: yaw rate (rad/s)
```

**`PredictedState`**: Future state with uncertainty
```python
- state: VehicleState
- covariance: 5×5 numpy array
- time_offset: seconds from current time
```

**`Vehicle`**: Main vehicle agent class
```python
- state: Current VehicleState
- P: 5×5 state covariance matrix
- predictions: List[PredictedState]
- collision_risks: Dict[vehicle_id → probability]
```

### Key Functions

**`bicycle_model_step()`**: Discrete kinematic bicycle model
```python
x_{k+1} = x_k + v_k * cos(ψ_k) * Δt
y_{k+1} = y_k + v_k * sin(ψ_k) * Δt
ψ_{k+1} = ψ_k + (v_k/L) * tan(δ_k) * Δt
v_{k+1} = v_k + a_k * Δt
```

**`propagate_covariance_simple()`**: Simplified UKF covariance propagation
- Currently: `P_new = P + Q`
- **TODO**: Replace with full UKF sigma point propagation

**`predict_trajectory()`**: Multi-step prediction with uncertainty
- Propagates state and covariance forward
- Returns list of `PredictedState` objects

**`compute_collision_probability_api_placeholder()`**: Collision probability calculation
- **Placeholder for your API integration**
- Currently uses simple Gaussian distance-based approximation
- Should implement b-plane projection method

**`draw_uncertainty_ellipse()`**: Visualization helper
- Computes eigenvalues/eigenvectors of 2×2 position covariance
- Draws ellipse as polygon

### Integration Points for Your API

Replace `compute_collision_probability_api_placeholder()` with your API call:

```python
def compute_collision_probability_api_placeholder(ego_pred: PredictedState, 
                                                   other_pred: PredictedState) -> float:
    """
    TODO: Replace with actual API call
    
    Expected inputs to API:
    - ego_state: [x, y, v, psi, psi_dot]
    - ego_covariance: 5×5 matrix
    - other_state: [x, y, v, psi, psi_dot]
    - other_covariance: 5×5 matrix
    
    Expected output:
    - collision_probability: float [0, 1]
    
    API should:
    1. Compute relative state: Δx = ego - other
    2. Compute combined covariance: P_Δ = P_ego + P_other
    3. Project to b-plane (perpendicular to relative velocity)
    4. Compute P(miss distance < collision footprint)
    """
    
    # Your API call here
    # result = api.compute_collision_prob(
    #     ego_pred.state.to_array(),
    #     ego_pred.covariance,
    #     other_pred.state.to_array(),
    #     other_pred.covariance
    # )
    # return result
    
    pass  # current placeholder implementation
```

## Next Steps / TODOs

### For Full UKF Implementation:
1. **Implement sigma point generation** (unscented transform)
2. **Replace `propagate_covariance_simple()`** with proper sigma point propagation
3. **Add measurement updates** (if you have sensor data)
4. **Tune process noise** (`Q`) and measurement noise (`R`) matrices

### For B-Plane Method:
1. **Integrate your collision probability API**
2. **Implement proper b-plane projection**
3. **Add relative velocity computation**
4. **Account for vehicle footprint/geometry**

### Enhancements:
1. Add vehicle-vehicle communication for cooperative awareness
2. Implement adaptive process noise based on driving conditions
3. Add road constraint awareness (vehicles shouldn't predict off-road)
4. Add visualization for collision zones/b-planes
5. Log collision events and statistics
6. Add scenario replay/recording

## Performance Notes

- **Current**: ~60 FPS with 20 vehicles, predictions enabled
- **Collision checks**: O(N²) for N vehicles, run every 0.2s
- **If slow**: Reduce `NUM_AGENTS`, increase `PREDICTION_UPDATE_INTERVAL`, or disable predictions (press P)

## Troubleshooting

**No vehicles appear**:
- Check that `wampus.json` exists and is valid
- Verify nodes have valid x/y coordinates

**Ellipses not showing**:
- Press `U` to toggle
- Check that covariance values are reasonable (not too small/large)

**Simulation crashes**:
- Check numpy/pygame versions
- Verify all vehicles have valid starting positions
- Check for NaN in covariance matrices

## References

This implementation is based on:
- Kinematic bicycle models (vehicle dynamics)
- Unscented Kalman Filter (state estimation)
- B-plane method (asteroid close-approach probability)
- Gaussian uncertainty propagation

See `plan.md` for detailed mathematical background.

