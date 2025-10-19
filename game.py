# collision_prediction_sim.py
# Vehicle simulation with UKF state estimation and collision prediction
# Based on bicycle model and celestial mechanics b-plane approach

import json, math, random, sys
from pathlib import Path
import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------- Display / Style ----------
WIN_W, WIN_H = 1200, 800
MARGIN = 60
BG = (240, 242, 245)  # Light gray background (like Google Maps)
DEFAULT_WIDTH, DEFAULT_COLOR = 3, (255, 255, 255)

# Realistic road styling with lane information
HIGHWAY_STYLE = {
    "motorway": {
        "width": 16, 
        "color": (255, 200, 100),  # Orange
        "outline": (200, 150, 50),
        "lanes": 4,
        "label_color": (100, 70, 20)
    },
    "trunk": {
        "width": 14, 
        "color": (255, 210, 110),
        "outline": (200, 160, 60),
        "lanes": 4,
        "label_color": (100, 75, 25)
    },
    "primary": {
        "width": 12, 
        "color": (255, 235, 150),  # Yellow
        "outline": (200, 180, 100),
        "lanes": 3,
        "label_color": (100, 85, 40)
    },
    "secondary": {
        "width": 10, 
        "color": (255, 255, 200),  # Light yellow
        "outline": (200, 200, 150),
        "lanes": 2,
        "label_color": (100, 100, 60)
    },
    "tertiary": {
        "width": 9, 
        "color": (255, 255, 255),  # White
        "outline": (180, 180, 180),
        "lanes": 2,
        "label_color": (80, 80, 80)
    },
    "residential": {
        "width": 8, 
        "color": (255, 255, 255),  # White
        "outline": (200, 200, 200),
        "lanes": 2,
        "label_color": (100, 100, 100)
    },
    "unclassified": {
        "width": 7, 
        "color": (250, 250, 250),
        "outline": (210, 210, 210),
        "lanes": 2,
        "label_color": (120, 120, 120)
    },
    "service": {
        "width": 5, 
        "color": (245, 245, 245),
        "outline": (220, 220, 220),
        "lanes": 1,
        "label_color": (140, 140, 140)
    },
    "living_street": {
        "width": 6, 
        "color": (248, 248, 248),
        "outline": (215, 215, 215),
        "lanes": 1,
        "label_color": (130, 130, 130)
    },
    "footway": {
        "width": 3, 
        "color": (220, 220, 220),
        "outline": (180, 180, 180),
        "lanes": 0,
        "label_color": (150, 150, 150)
    },
    "path": {
        "width": 3, 
        "color": (215, 215, 215),
        "outline": (175, 175, 175),
        "lanes": 0,
        "label_color": (150, 150, 150)
    },
}
ARROW_LEN = 14  # px
ARROW_ANG = math.radians(22)

# ---------- Simulation Parameters ----------
NUM_AGENTS = 20
SPEED_MIN = 3.0   # m/s (slower vehicles)
SPEED_MAX = 15.0  # m/s (faster vehicles - more variety)
AGENT_COLORS = [(80, 200, 255), (255, 140, 90), (180, 255, 140), (250, 220, 80)]

# Lane parameters
LANE_WIDTH = 3.5  # meters (standard lane width)
USE_LANES = True  # Enable lane-based positioning

# Vehicle parameters (bicycle model)
VEHICLE_LENGTH = 4.5  # m (wheelbase L)
VEHICLE_WIDTH = 2.0   # m
MAX_STEERING_ANGLE = math.radians(30)  # rad

# UKF / Prediction parameters
DT = 0.1  # prediction time step (s)
PREDICTION_HORIZON = 5.0  # predict out to 5 seconds
NUM_PRED_STEPS = int(PREDICTION_HORIZON / DT)

# Process noise (tuning parameters for UKF)
PROCESS_NOISE_POS = 0.5  # m
PROCESS_NOISE_VEL = 0.2  # m/s
PROCESS_NOISE_HEADING = 0.05  # rad
PROCESS_NOISE_YAW_RATE = 0.02  # rad/s

# Visualization
SHOW_UNCERTAINTY = True
SHOW_PREDICTIONS = True
SHOW_LABELS = True  # Show vehicle IDs and road names
SHOW_LANE_MARKINGS = True  # Draw lane dividers on roads
ELLIPSE_SIGMA = 2.0  # draw 2-sigma ellipses
COLLISION_THRESHOLD = 0.01  # probability threshold for warning

# ---------- Data IO ----------
def load_graph(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["nodes"], data["edges"]

def load_buildings(path: Path):
    """Load building polygons from JSON"""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

# ---------- Projection ----------
def compute_bounds(nodes: dict):
    lons = [n["x"] for n in nodes.values()]
    lats = [n["y"] for n in nodes.values()]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    if abs(max_lon - min_lon) < 1e-12: max_lon += 1e-12
    if abs(max_lat - min_lat) < 1e-12: max_lat += 1e-12
    return min_lon, max_lon, min_lat, max_lat

def make_projector(nodes: dict, win_w: int, win_h: int, margin: int):
    min_lon, max_lon, min_lat, max_lat = compute_bounds(nodes)
    usable_w = max(1, win_w - 2 * margin)
    usable_h = max(1, win_h - 2 * margin)
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat
    s = min(usable_w / span_lon, usable_h / span_lat)
    offx = margin + (usable_w - s * span_lon) * 0.5
    offy = margin + (usable_h - s * span_lat) * 0.5

    def world_to_screen(lon: float, lat: float, scale=1.0, pan=(0, 0)):
        x = offx + pan[0] + (lon - min_lon) * s * scale
        y = offy + pan[1] + (max_lat - lat) * s * scale  # invert Y
        return int(round(x)), int(round(y))

    return world_to_screen, (min_lon, max_lon, min_lat, max_lat), s, (offx, offy)

def highway_style(hwy):
    """Get road style dictionary"""
    default = {
        "width": DEFAULT_WIDTH,
        "color": DEFAULT_COLOR,
        "outline": (180, 180, 180),
        "lanes": 1,
        "label_color": (100, 100, 100)
    }
    return HIGHWAY_STYLE.get(hwy, default)

# ---------- State Vector & UKF Helpers ----------
@dataclass
class VehicleState:
    """State vector: [x, y, v, psi, psi_dot]
    x, y: position (m)
    v: speed (m/s)
    psi: heading angle (rad)
    psi_dot: yaw rate (rad/s)
    """
    x: float
    y: float
    v: float
    psi: float
    psi_dot: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.v, self.psi, self.psi_dot])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'VehicleState':
        return VehicleState(arr[0], arr[1], arr[2], arr[3], arr[4])

@dataclass
class PredictedState:
    """Predicted state with uncertainty"""
    state: VehicleState
    covariance: np.ndarray  # 5x5 state covariance
    time_offset: float  # seconds from current time

def bicycle_model_step(state: VehicleState, delta: float, accel: float, dt: float, L: float) -> VehicleState:
    """Discrete kinematic bicycle model
    delta: steering angle (rad)
    accel: acceleration (m/s^2)
    dt: time step (s)
    L: wheelbase length (m)
    """
    x_new = state.x + state.v * math.cos(state.psi) * dt
    y_new = state.y + state.v * math.sin(state.psi) * dt
    v_new = state.v + accel * dt
    psi_new = state.psi + (state.v / L) * math.tan(delta) * dt
    psi_dot_new = (state.v / L) * math.tan(delta)
    
    # Normalize heading to [-pi, pi]
    while psi_new > math.pi: psi_new -= 2 * math.pi
    while psi_new < -math.pi: psi_new += 2 * math.pi
    
    return VehicleState(x_new, y_new, v_new, psi_new, psi_dot_new)

def propagate_covariance_simple(P: np.ndarray, dt: float) -> np.ndarray:
    """Simplified covariance propagation (placeholder for full UKF)
    In reality, this would use Jacobian or sigma points
    """
    # Add process noise
    Q = np.diag([
        PROCESS_NOISE_POS**2,
        PROCESS_NOISE_POS**2,
        PROCESS_NOISE_VEL**2,
        PROCESS_NOISE_HEADING**2,
        PROCESS_NOISE_YAW_RATE**2
    ]) * dt
    
    # Simple propagation: P_new ≈ P + Q (actual UKF would use F*P*F' + Q)
    return P + Q

def predict_trajectory(state: VehicleState, P: np.ndarray, delta: float, accel: float, 
                       n_steps: int, dt: float, L: float) -> List[PredictedState]:
    """Predict vehicle trajectory with uncertainty over n_steps"""
    predictions = []
    current_state = state
    current_P = P.copy()
    
    for i in range(n_steps):
        # Propagate state
        current_state = bicycle_model_step(current_state, delta, accel, dt, L)
        # Propagate covariance
        current_P = propagate_covariance_simple(current_P, dt)
        
        predictions.append(PredictedState(
            state=current_state,
            covariance=current_P.copy(),
            time_offset=(i + 1) * dt
        ))
    
    return predictions

def compute_collision_probability_api_placeholder(ego_pred: PredictedState, 
                                                   other_pred: PredictedState) -> float:
    """Placeholder for API call to compute collision probability using b-plane method
    
    In production, this would:
    1. Compute relative state: delta_x = ego - other
    2. Compute combined covariance: P_delta = P_ego + P_other
    3. Project to b-plane (plane perpendicular to relative velocity)
    4. Compute probability that miss-distance falls within collision footprint
    
    For now, return simple distance-based estimate
    """
    dx = ego_pred.state.x - other_pred.state.x
    dy = ego_pred.state.y - other_pred.state.y
    dist = math.sqrt(dx**2 + dy**2)
    
    # Simple Gaussian approximation based on distance and uncertainty
    # Extract position uncertainties (diagonal elements)
    ego_std = math.sqrt(ego_pred.covariance[0, 0] + ego_pred.covariance[1, 1])
    other_std = math.sqrt(other_pred.covariance[0, 0] + other_pred.covariance[1, 1])
    combined_std = math.sqrt(ego_std**2 + other_std**2)
    
    # Collision if within ~2 vehicle widths
    collision_radius = 2 * VEHICLE_WIDTH
    
    if combined_std < 1e-6:
        return 1.0 if dist < collision_radius else 0.0
    
    # Gaussian CDF approximation
    z = (dist - collision_radius) / combined_std
    if z < -3:
        return 1.0
    elif z > 3:
        return 0.0
    else:
        # Rough approximation of collision probability
        return max(0.0, min(1.0, math.exp(-z**2 / 2)))

def draw_uncertainty_ellipse(surf, center_x: int, center_y: int, 
                             P_2x2: np.ndarray, sigma: float, color, scale: float):
    """Draw uncertainty ellipse from 2x2 position covariance matrix
    Uses eigenvalues/eigenvectors to get principal axes
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P_2x2)
        # Handle numerical issues
        if np.any(eigenvalues < 0):
            return
        
        # Semi-major and semi-minor axes (scaled by sigma)
        width = 2 * sigma * math.sqrt(max(eigenvalues[0], 0)) * scale
        height = 2 * sigma * math.sqrt(max(eigenvalues[1], 0)) * scale
        
        # Angle of major axis
        angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Draw ellipse (pygame doesn't have ellipse rotation, so approximate with polygon)
        points = []
        n_points = 32
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            # Point on unit circle
            x_local = math.cos(theta) * width / 2
            y_local = math.sin(theta) * height / 2
            # Rotate
            x_rot = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_rot = x_local * math.sin(angle) + y_local * math.cos(angle)
            # Translate
            points.append((center_x + x_rot, center_y - y_rot))
        
        if len(points) > 2:
            pygame.draw.polygon(surf, color, points, 1)
    except:
        pass  # Skip if ellipse computation fails

# ---------- Road Graph ----------
class RoadGraph:
    def __init__(self, nodes: dict, edges: list[dict]):
        self.nodes = nodes
        self.edges = edges
        self.out = {}  # node_id -> list of edges starting here (directed)
        for e in edges:
            self.out.setdefault(e["from"], []).append(e)

        lats = [n["y"] for n in nodes.values()]
        self.ref_lat = sum(lats)/len(lats)
        self.m_per_deg_lat = 110_540.0
        self.m_per_deg_lon = 111_320.0 * math.cos(math.radians(self.ref_lat))

    def lonlat_to_m(self, lon, lat):
        return lon * self.m_per_deg_lon, lat * self.m_per_deg_lat

    def node_xy_m(self, nid: str):
        n = self.nodes[nid]
        return self.lonlat_to_m(n["x"], n["y"])

    def interp_lonlat(self, u: str, v: str, t: float):
        a, b = self.nodes[u], self.nodes[v]
        lon = a["x"] + (b["x"] - a["x"]) * t
        lat = a["y"] + (b["y"] - a["y"]) * t
        return lon, lat

    def heading_rad(self, u: str, v: str):
        ax, ay = self.node_xy_m(u)
        bx, by = self.node_xy_m(v)
        return math.atan2(by - ay, bx - ax)

# ---------- Vehicle Agents with UKF State Estimation ----------
class Vehicle:
    def __init__(self, G: RoadGraph, vehicle_id: int):
        self.id = vehicle_id
        self.G = G
        self.color = AGENT_COLORS[vehicle_id % len(AGENT_COLORS)]
        
        # Choose a random starting edge
        candidates = [e for e in G.edges if float(e.get("length_m", 0.0)) > 0.1]
        self.edge = random.choice(candidates)
        self.u = self.edge["from"]
        self.v = self.edge["to"]
        self.t = random.random()
        self.prev_u = None
        
        # Lane selection - based on number of lanes available
        num_lanes = self.edge.get("lanes", "1")
        try:
            if isinstance(num_lanes, list):
                num_lanes = int(num_lanes[0]) if num_lanes else 1
            elif isinstance(num_lanes, str):
                num_lanes = int(num_lanes)
            else:
                num_lanes = int(num_lanes)
        except:
            num_lanes = 1
        
        # Ensure valid lane count
        num_lanes = max(1, num_lanes)
        self.num_lanes = num_lanes
        
        # Choose a lane (0 = rightmost, n-1 = leftmost)
        # For right-hand traffic: right lane is default, left for passing
        if num_lanes > 1:
            self.lane = random.randint(0, num_lanes - 1)
        else:
            self.lane = 0
        
        # Initialize state vector
        lon, lat = G.interp_lonlat(self.u, self.v, self.t)
        x, y = G.lonlat_to_m(lon, lat)
        heading = G.heading_rad(self.u, self.v)
        
        # More varied speeds - create speed "classes"
        speed_class = random.random()
        if speed_class < 0.3:  # 30% slow vehicles
            speed = random.uniform(SPEED_MIN, SPEED_MIN + 3.0)
        elif speed_class < 0.7:  # 40% medium vehicles
            speed = random.uniform(SPEED_MIN + 3.0, SPEED_MAX - 3.0)
        else:  # 30% fast vehicles
            speed = random.uniform(SPEED_MAX - 3.0, SPEED_MAX)
        
        self.state = VehicleState(
            x=x, y=y, v=speed, 
            psi=heading, psi_dot=0.0
        )
        
        # Initialize state covariance (5x5 matrix)
        self.P = np.diag([
            1.0,  # x variance (m^2)
            1.0,  # y variance (m^2)
            0.5,  # v variance (m^2/s^2)
            0.1,  # psi variance (rad^2)
            0.05  # psi_dot variance (rad^2/s^2)
        ])
        
        # Control inputs (simple controller)
        self.target_speed = speed
        self.steering_angle = 0.0
        
        # Predicted trajectory
        self.predictions: List[PredictedState] = []
        self.collision_risks: dict = {}  # vehicle_id -> max probability

    def _get_lane_offset(self, heading: float) -> Tuple[float, float]:
        """Calculate lateral offset in meters for the current lane
        Returns (dx, dy) offset from road center
        """
        if not USE_LANES or self.num_lanes <= 1:
            return (0.0, 0.0)
        
        # Calculate offset from center of road
        # Lane 0 is rightmost, lane n-1 is leftmost
        # Offset from center = (lane - (num_lanes-1)/2) * LANE_WIDTH
        lane_offset = (self.lane - (self.num_lanes - 1) / 2.0) * LANE_WIDTH
        
        # Perpendicular to road direction (right is positive offset)
        # For heading angle, perpendicular is heading - 90 degrees
        perp_angle = heading - math.pi / 2
        
        dx = lane_offset * math.cos(perp_angle)
        dy = lane_offset * math.sin(perp_angle)
        
        return (dx, dy)

    def _choose_next_edge(self):
        """Same logic as before for choosing next edge"""
        outs = self.G.out.get(self.u, [])
        if not outs:
            if self.prev_u:
                back = [e for e in outs if e["to"] == self.prev_u]
                if back:
                    return back[0]
            return self.edge

        if self.prev_u is None:
            base_ang = self.G.heading_rad(self.u, outs[0]["to"])
        else:
            base_ang = self.G.heading_rad(self.prev_u, self.u)

        cands = []
        for e in outs:
            ang = self.G.heading_rad(self.u, e["to"])
            d = ang - base_ang
            while d <= -math.pi: d += 2*math.pi
            while d >  math.pi: d -= 2*math.pi
            cands.append((abs(d), d, e))

        eps = 1e-3
        weights = []
        for (absd, d, e) in cands:
            w = 1.0 / (absd + eps)
            weights.append(max(w, 0.05))
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for w, trip in zip(weights, cands):
            acc += w
            if r <= acc:
                return trip[2]
        return cands[0][2]
    
    def _update_lane_for_new_edge(self):
        """Update lane selection when transitioning to a new edge"""
        # Get number of lanes on new edge
        num_lanes = self.edge.get("lanes", "1")
        try:
            if isinstance(num_lanes, list):
                num_lanes = int(num_lanes[0]) if num_lanes else 1
            elif isinstance(num_lanes, str):
                num_lanes = int(num_lanes)
            else:
                num_lanes = int(num_lanes)
        except:
            num_lanes = 1
        
        # Ensure valid lane count
        num_lanes = max(1, num_lanes)
        self.num_lanes = num_lanes
        
        # If new road has fewer lanes, adjust lane number
        if self.lane >= self.num_lanes:
            self.lane = self.num_lanes - 1
        
        # Occasionally change lanes (20% chance)
        if random.random() < 0.2 and self.num_lanes > 1:
            self.lane = random.randint(0, self.num_lanes - 1)
    
    def _compute_control_inputs(self):
        """Simple controller to follow road and maintain speed"""
        # Speed control
        accel = (self.target_speed - self.state.v) * 0.5  # proportional controller
        accel = np.clip(accel, -3.0, 2.0)  # reasonable accel/brake limits
        
        # Steering: align with road heading
        target_heading = self.G.heading_rad(self.u, self.v)
        heading_error = target_heading - self.state.psi
        # Normalize to [-pi, pi]
        while heading_error > math.pi: heading_error -= 2*math.pi
        while heading_error < -math.pi: heading_error += 2*math.pi
        
        self.steering_angle = np.clip(heading_error * 2.0, 
                                      -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        
        return self.steering_angle, accel

    def update(self, dt: float):
        """Update vehicle state using bicycle model, constrained to roads"""
        # Get control inputs
        delta, accel = self._compute_control_inputs()
        
        # Update state using bicycle model
        self.state = bicycle_model_step(self.state, delta, accel, dt, VEHICLE_LENGTH)
        
        # Update covariance (simplified - real UKF would use sigma points)
        self.P = propagate_covariance_simple(self.P, dt)
        
        # CONSTRAIN TO ROAD: Update position along edge (for road following)
        L = float(self.edge.get("length_m", 1.0))
        L = max(L, 1e-6)
        self.t += (self.state.v * dt) / L
        
        # Handle edge transitions
        if self.t >= 1.0:
            self.t = 0.0
            self.prev_u = self.u
            self.u = self.v
            self.edge = self._choose_next_edge()
            self.v = self.edge["to"]
            # Update lane for new edge
            self._update_lane_for_new_edge()
            # Occasionally update target speed for variety (less often now for consistency)
            if random.random() < 0.3:
                speed_class = random.random()
                if speed_class < 0.3:
                    self.target_speed = random.uniform(SPEED_MIN, SPEED_MIN + 3.0)
                elif speed_class < 0.7:
                    self.target_speed = random.uniform(SPEED_MIN + 3.0, SPEED_MAX - 3.0)
                else:
                    self.target_speed = random.uniform(SPEED_MAX - 3.0, SPEED_MAX)

        # SYNC: Force vehicle state position to match road position with lane offset
        # This ensures vehicles always appear on roads despite bicycle model drift
        lon, lat = self.G.interp_lonlat(self.u, self.v, self.t)
        road_x, road_y = self.G.lonlat_to_m(lon, lat)
        road_heading = self.G.heading_rad(self.u, self.v)
        
        # Apply lane offset
        lane_dx, lane_dy = self._get_lane_offset(road_heading)
        road_x += lane_dx
        road_y += lane_dy
        
        # Update state to match road (with some smoothing for realism)
        blend = 0.7  # How much to trust road vs. bicycle model
        self.state.x = blend * road_x + (1 - blend) * self.state.x
        self.state.y = blend * road_y + (1 - blend) * self.state.y
        self.state.psi = blend * road_heading + (1 - blend) * self.state.psi
    
    def predict_future(self):
        """Predict future trajectory using current state and simple assumptions"""
        delta, accel = self._compute_control_inputs()
        self.predictions = predict_trajectory(
            self.state, self.P, delta, accel,
            NUM_PRED_STEPS, DT, VEHICLE_LENGTH
        )
    
    def m_to_lonlat(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert meters back to lon/lat"""
        lon = x_m / self.G.m_per_deg_lon
        lat = y_m / self.G.m_per_deg_lat
        return lon, lat
    
    def get_position_lonlat(self) -> Tuple[float, float, float]:
        """Get current position in lon/lat and heading"""
        lon, lat = self.m_to_lonlat(self.state.x, self.state.y)
        return lon, lat, self.state.psi

    def draw(self, surf, world_to_screen, scale, pan, base_scale, font):
        """Draw vehicle with uncertainty ellipse, predictions, and label"""
        lon, lat, ang = self.get_position_lonlat()
        px, py = world_to_screen(lon, lat, scale=scale, pan=pan)
        
        # Determine color based on collision risk
        color = self.color
        is_high_risk = False
        if self.collision_risks:
            max_risk = max(self.collision_risks.values())
            if max_risk > COLLISION_THRESHOLD:
                is_high_risk = True
                # Interpolate to red based on risk
                risk_factor = min(max_risk / 0.5, 1.0)  # scale to [0, 1]
                color = (
                    int(255),
                    int(50 + self.color[1] * (1 - risk_factor) * 0.5),
                    int(50 + self.color[2] * (1 - risk_factor) * 0.5)
                )
        
        # Draw vehicle as arrow with outline for better visibility
        tip = (px, py)
        tail = (px - ARROW_LEN * math.cos(ang), py + ARROW_LEN * math.sin(ang))
        left = (tip[0] - ARROW_LEN * math.cos(ang - ARROW_ANG),
                tip[1] + ARROW_LEN * math.sin(ang - ARROW_ANG))
        right = (tip[0] - ARROW_LEN * math.cos(ang + ARROW_ANG),
                 tip[1] + ARROW_LEN * math.sin(ang + ARROW_ANG))
        
        # Draw outline (black)
        outline_width = 6 if is_high_risk else 5
        pygame.draw.line(surf, (40, 40, 50), tail, tip, outline_width)
        pygame.draw.line(surf, (40, 40, 50), tip, left, outline_width)
        pygame.draw.line(surf, (40, 40, 50), tip, right, outline_width)
        
        # Draw vehicle body
        line_width = 4 if is_high_risk else 3
        pygame.draw.line(surf, color, tail, tip, line_width)
        pygame.draw.line(surf, color, tip, left, line_width)
        pygame.draw.line(surf, color, tip, right, line_width)
        
        # Draw vehicle ID label with speed
        if SHOW_LABELS:
            # Convert speed from m/s to km/h for display
            speed_kmh = self.state.v * 3.6
            label_text = font.render(f"V{self.id} ({speed_kmh:.0f}km/h)", True, (40, 40, 50))
            label_rect = label_text.get_rect(center=(px, py - 22))
            
            # Background for label
            bg_rect = label_rect.inflate(6, 3)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.fill((255, 255, 255))
            bg_surface.set_alpha(220)
            surf.blit(bg_surface, bg_rect)
            
            surf.blit(label_text, label_rect)
            
            # Draw lane indicator (small dot)
            if USE_LANES and self.num_lanes > 1:
                lane_color = (100, 150, 200)
                lane_indicator_y = py - 35
                for i in range(self.num_lanes):
                    dot_x = px - (self.num_lanes - 1) * 3 + i * 6
                    if i == self.lane:
                        pygame.draw.circle(surf, lane_color, (dot_x, lane_indicator_y), 3)
                    else:
                        pygame.draw.circle(surf, (180, 180, 180), (dot_x, lane_indicator_y), 2)
        
        # Draw current uncertainty ellipse
        if SHOW_UNCERTAINTY:
            P_2x2 = self.P[:2, :2]  # extract x,y covariance
            draw_uncertainty_ellipse(surf, px, py, P_2x2, ELLIPSE_SIGMA, color, base_scale * scale)
        
        # Draw predicted trajectory
        if SHOW_PREDICTIONS and self.predictions:
            for i, pred in enumerate(self.predictions[::5]):  # every 5th prediction to reduce clutter
                pred_lon, pred_lat = self.m_to_lonlat(pred.state.x, pred.state.y)
                pred_px, pred_py = world_to_screen(pred_lon, pred_lat, scale=scale, pan=pan)
                
                # Draw small dot
                alpha = 1.0 - (i * 5 / len(self.predictions))  # fade out over time
                dot_color = tuple(int(c * alpha) for c in color)
                pygame.draw.circle(surf, dot_color, (pred_px, pred_py), 3)
                
                # Draw uncertainty ellipse at prediction
                P_pred_2x2 = pred.covariance[:2, :2]
                ellipse_color = tuple(int(c * alpha * 0.5) for c in color)
                draw_uncertainty_ellipse(surf, pred_px, pred_py, P_pred_2x2, 
                                        ELLIPSE_SIGMA, ellipse_color, base_scale * scale)

# ---------- Drawing helpers ----------
def draw_buildings(screen, buildings, w2s, scale, pan):
    """Draw building footprints as realistic structures"""
    building_fill = (200, 200, 210)  # Light gray
    building_outline = (140, 140, 150)  # Darker outline
    
    for building in buildings:
        if building['type'] == 'Polygon':
            coords = building['coordinates']
            if len(coords) < 3:
                continue
            
            # Convert to screen coordinates
            screen_coords = []
            for lon, lat in coords:
                px, py = w2s(lon, lat, scale=scale, pan=pan)
                screen_coords.append((px, py))
            
            if len(screen_coords) >= 3:
                try:
                    # Draw filled building
                    pygame.draw.polygon(screen, building_fill, screen_coords, 0)
                    # Draw outline
                    pygame.draw.polygon(screen, building_outline, screen_coords, 2)
                except:
                    pass  # Skip invalid polygons

def draw_edges(screen, nodes, edges, w2s, scale, pan, base_scale, zoom_scale, font):
    """Draw roads with realistic styling, lane markings, and labels"""
    drawn_labels = {}  # Track which labels we've drawn to avoid duplicates
    
    for e in edges:
        u, v = e["from"], e["to"]
        if u not in nodes or v not in nodes: 
            continue
            
        a, b = nodes[u], nodes[v]
        p0 = w2s(a["x"], a["y"], scale=zoom_scale, pan=pan)
        p1 = w2s(b["x"], b["y"], scale=zoom_scale, pan=pan)
        
        style = highway_style(e.get("highway"))
        width = style["width"]
        color = style["color"]
        outline = style["outline"]
        lanes = style["lanes"]
        
        # Draw road outline (darker border)
        pygame.draw.line(screen, outline, p0, p1, width + 4)
        
        # Draw road surface
        pygame.draw.line(screen, color, p0, p1, width)
        
        # Draw lane markings for multi-lane roads
        if SHOW_LANE_MARKINGS and lanes >= 2 and width >= 8:
            # Calculate perpendicular offset for lane dividers
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Unit perpendicular vector
                perp_x = -dy / length
                perp_y = dx / length
                
                # Draw center line (dashed)
                num_dashes = int(length / 20)
                for i in range(num_dashes):
                    t1 = i / num_dashes
                    t2 = (i + 0.5) / num_dashes
                    dash_start = (int(p0[0] + dx * t1), int(p0[1] + dy * t1))
                    dash_end = (int(p0[0] + dx * t2), int(p0[1] + dy * t2))
                    pygame.draw.line(screen, (200, 200, 120), dash_start, dash_end, 2)
        
        # Draw road name labels (if available and SHOW_LABELS enabled)
        if SHOW_LABELS and e.get("name"):
            road_name = e["name"]
            # Handle case where name might be a list
            if isinstance(road_name, list):
                road_name = road_name[0] if road_name else None
            
            # Only draw valid string names
            if road_name and isinstance(road_name, str):
                # Only draw each road name once
                if road_name not in drawn_labels:
                    mx = (p0[0] + p1[0]) // 2
                    my = (p0[1] + p1[1]) // 2
                    
                    # Create label with background
                    label_text = font.render(road_name, True, style["label_color"])
                    label_rect = label_text.get_rect(center=(mx, my - 12))
                    
                    # Draw white background for readability
                    bg_rect = label_rect.inflate(8, 4)
                    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
                    bg_surface.fill((255, 255, 255))
                    bg_surface.set_alpha(200)
                    screen.blit(bg_surface, bg_rect)
                    
                    screen.blit(label_text, label_rect)
                    drawn_labels[road_name] = True
        
        # Draw directional arrows for one-way streets
        if e.get("oneway") is True:
            mx = (p0[0] + p1[0]) / 2.0
            my = (p0[1] + p1[1]) / 2.0
            ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            
            arrow_color = (100, 100, 120)
            left = (mx - 8 * math.cos(ang - math.radians(30)),
                    my - 8 * math.sin(ang - math.radians(30)))
            right = (mx - 8 * math.cos(ang + math.radians(30)),
                     my - 8 * math.sin(ang + math.radians(30)))
            tip = (mx + 4 * math.cos(ang), my + 4 * math.sin(ang))
            
            pygame.draw.line(screen, arrow_color, left, tip, 2)
            pygame.draw.line(screen, arrow_color, right, tip, 2)

# ---------- Collision Detection ----------
def compute_all_collision_probabilities(vehicles: List[Vehicle]):
    """Compute collision probabilities between all vehicle pairs"""
    n = len(vehicles)
    
    # Reset collision risks
    for v in vehicles:
        v.collision_risks = {}
    
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = vehicles[i], vehicles[j]
            
            # Skip if either has no predictions
            if not v1.predictions or not v2.predictions:
                continue
            
            # Check collision probability at each time step
            max_prob = 0.0
            for pred1, pred2 in zip(v1.predictions, v2.predictions):
                prob = compute_collision_probability_api_placeholder(pred1, pred2)
                max_prob = max(max_prob, prob)
            
            # Store collision risk
            if max_prob > 0.001:  # only store non-negligible risks
                v1.collision_risks[v2.id] = max_prob
                v2.collision_risks[v1.id] = max_prob

# ---------- Main ----------
def main():
    p = Path("wampus.json")
    if not p.exists():
        print("ERROR: Save your JSON as wampus.json next to this file.")
        sys.exit(1)
    nodes, edges = load_graph(p)
    
    # Load building data (optional - will be empty list if not available)
    buildings = load_buildings(Path("buildings.json"))
    print(f"Loaded {len(buildings)} buildings")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Collision Prediction Simulation — Right-drag: pan | Wheel: zoom | R: reset | U: uncertainty | P: predictions | B: buildings")
    clock = pygame.time.Clock()

    w2s, bounds, base_scale, (base_offx, base_offy) = make_projector(nodes, WIN_W, WIN_H, MARGIN)

    scale = 1.0
    pan = [0.0, 0.0]
    dragging = False
    last_mouse = (0, 0)
    show_buildings = len(buildings) > 0  # Show buildings by default if available

    G = RoadGraph(nodes, edges)
    vehicles = [Vehicle(G, i) for i in range(NUM_AGENTS)]
    
    # Prediction update counter (don't predict every frame)
    prediction_timer = 0.0
    PREDICTION_UPDATE_INTERVAL = 0.2  # seconds

    running = True
    frame_count = 0
    while running:
        dt = clock.tick(60) / 1000.0
        frame_count += 1

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 3:
                    dragging = True
                    last_mouse = ev.pos
                elif ev.button == 4:
                    scale *= 1.1
                elif ev.button == 5:
                    scale = max(0.1, scale / 1.1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 3:
                    dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                mx, my = ev.pos
                lx, ly = last_mouse
                pan[0] += (mx - lx)
                pan[1] += (my - ly)
                last_mouse = ev.pos
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    scale = 1.0; pan = [0.0, 0.0]
                elif ev.key == pygame.K_u:
                    global SHOW_UNCERTAINTY
                    SHOW_UNCERTAINTY = not SHOW_UNCERTAINTY
                elif ev.key == pygame.K_p:
                    global SHOW_PREDICTIONS
                    SHOW_PREDICTIONS = not SHOW_PREDICTIONS
                elif ev.key == pygame.K_b:
                    show_buildings = not show_buildings
                elif ev.key == pygame.K_l:
                    global SHOW_LABELS
                    SHOW_LABELS = not SHOW_LABELS
                elif ev.key == pygame.K_n:
                    global USE_LANES
                    USE_LANES = not USE_LANES

        # Update vehicles
        for v in vehicles:
            v.update(dt)

        # Update predictions and collision probabilities periodically
        prediction_timer += dt
        if prediction_timer >= PREDICTION_UPDATE_INTERVAL:
            prediction_timer = 0.0
            for v in vehicles:
                v.predict_future()
            compute_all_collision_probabilities(vehicles)

        # Draw
        screen.fill(BG)
        
        # Draw buildings first (background layer)
        if show_buildings and buildings:
            draw_buildings(screen, buildings, w2s, scale, pan)
        
        # Draw roads on top of buildings
        font_small = pygame.font.SysFont("Arial", 11, bold=True)
        draw_edges(screen, nodes, edges, w2s, scale, pan, base_scale, scale, font_small)
        
        # Draw vehicles on top
        for v in vehicles:
            v.draw(screen, w2s, scale, pan, base_scale, font_small)

        # ========== Professional UI Panel ==========
        panel_font = pygame.font.SysFont("Arial", 13)
        title_font = pygame.font.SysFont("Arial", 15, bold=True)
        
        # Top control panel
        panel_height = 70
        panel_surf = pygame.Surface((WIN_W, panel_height))
        panel_surf.fill((30, 35, 45))
        panel_surf.set_alpha(240)
        screen.blit(panel_surf, (0, 0))
        
        # Title
        title = title_font.render("Vehicle Collision Prediction Simulation", True, (200, 220, 240))
        screen.blit(title, (15, 10))
        
        # Controls
        controls = panel_font.render("Controls: Right-drag=Pan | Wheel=Zoom | R=Reset | U=Uncertainty | P=Predictions | B=Buildings | L=Labels | N=Lanes", True, (160, 175, 195))
        screen.blit(controls, (15, 32))
        
        # Status indicators
        status_items = []
        status_items.append(f"Vehicles: {NUM_AGENTS}")
        status_items.append(f"Uncertainty: {'✓' if SHOW_UNCERTAINTY else '✗'}")
        status_items.append(f"Predictions: {'✓' if SHOW_PREDICTIONS else '✗'}")
        status_items.append(f"Labels: {'✓' if SHOW_LABELS else '✗'}")
        status_items.append(f"Lanes: {'✓' if USE_LANES else '✗'}")
        status_items.append(f"Buildings: {'✓' if show_buildings else '✗'}")
        status_items.append(f"FPS: {int(clock.get_fps())}")
        
        status_text = " | ".join(status_items)
        status_render = panel_font.render(status_text, True, (140, 160, 180))
        screen.blit(status_render, (15, 50))
        
        # Collision warnings panel (right side)
        high_risk_pairs = []
        for v in vehicles:
            for other_id, prob in v.collision_risks.items():
                if prob > COLLISION_THRESHOLD and v.id < other_id:  # avoid duplicates
                    high_risk_pairs.append((v.id, other_id, prob))
        
        if high_risk_pairs:
            # Sort by probability (highest first)
            high_risk_pairs.sort(key=lambda x: x[2], reverse=True)
            
            warning_panel_width = 260
            warning_panel_height = min(150, 30 + len(high_risk_pairs[:5]) * 24)
            warning_surf = pygame.Surface((warning_panel_width, warning_panel_height))
            warning_surf.fill((45, 30, 30))
            warning_surf.set_alpha(240)
            screen.blit(warning_surf, (WIN_W - warning_panel_width - 15, 85))
            
            warning_title = title_font.render("⚠ Collision Warnings", True, (255, 220, 220))
            screen.blit(warning_title, (WIN_W - warning_panel_width - 10, 90))
            
            y_offset = 115
            warning_font = pygame.font.SysFont("Arial", 12, bold=True)
            for v1_id, v2_id, prob in high_risk_pairs[:5]:  # show top 5
                risk_color = (255, 100, 100) if prob > 0.5 else (255, 180, 100)
                msg = f"V{v1_id} ↔ V{v2_id}: {prob:.1%}"
                warning_render = warning_font.render(msg, True, risk_color)
                screen.blit(warning_render, (WIN_W - warning_panel_width - 5, y_offset))
                y_offset += 24
        
        # Legend panel (bottom right)
        legend_width = 200
        legend_height = 120
        legend_surf = pygame.Surface((legend_width, legend_height))
        legend_surf.fill((35, 40, 50))
        legend_surf.set_alpha(230)
        screen.blit(legend_surf, (WIN_W - legend_width - 15, WIN_H - legend_height - 15))
        
        legend_font = pygame.font.SysFont("Arial", 11)
        legend_y = WIN_H - legend_height - 10
        
        legend_title = panel_font.render("Legend", True, (200, 220, 240))
        screen.blit(legend_title, (WIN_W - legend_width - 10, legend_y))
        legend_y += 20
        
        # Road types
        road_label = legend_font.render("Road Types:", True, (180, 190, 200))
        screen.blit(road_label, (WIN_W - legend_width - 8, legend_y))
        legend_y += 18
        
        pygame.draw.line(screen, (255, 200, 100), (WIN_W - legend_width - 5, legend_y + 7), (WIN_W - legend_width + 15, legend_y + 7), 3)
        text = legend_font.render(" Motorway", True, (180, 190, 200))
        screen.blit(text, (WIN_W - legend_width + 20, legend_y + 2))
        legend_y += 18
        
        pygame.draw.line(screen, (255, 235, 150), (WIN_W - legend_width - 5, legend_y + 7), (WIN_W - legend_width + 15, legend_y + 7), 3)
        text = legend_font.render(" Primary", True, (180, 190, 200))
        screen.blit(text, (WIN_W - legend_width + 20, legend_y + 2))
        legend_y += 18
        
        pygame.draw.line(screen, (255, 255, 255), (WIN_W - legend_width - 5, legend_y + 7), (WIN_W - legend_width + 15, legend_y + 7), 3)
        text = legend_font.render(" Residential", True, (180, 190, 200))
        screen.blit(text, (WIN_W - legend_width + 20, legend_y + 2))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
