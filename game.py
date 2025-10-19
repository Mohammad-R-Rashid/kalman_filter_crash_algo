# collision_prediction_sim.py
# Vehicle simulation demonstrating celestial mechanics b-plane collision prediction
# Uses vehicle_algo.py for UKF state estimation and collision detection

import json, math, random, sys
from pathlib import Path
import pygame
import numpy as np
from typing import List, Tuple
from vehicle_algo import vehicle as VehicleAlgo, vehicle_simulation

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
NUM_AGENTS = 10
SPEED_MIN = 3.0   # m/s (slower vehicles)
SPEED_MAX = 15.0  # m/s (faster vehicles - more variety)

# Consistent color palette: teal (safe), amber (caution), red (danger)
COLOR_SAFE = (80, 200, 180)      # Teal
COLOR_CAUTION = (255, 180, 80)   # Amber
COLOR_DANGER = (255, 100, 100)   # Red
AGENT_COLORS = [COLOR_SAFE, (120, 180, 220), (160, 200, 160), (200, 200, 120)]

# Lane parameters
LANE_WIDTH = 3.5  # meters (standard lane width)
USE_LANES = True  # Enable lane-based positioning

# Vehicle parameters (bicycle model)
VEHICLE_LENGTH = 2.5  # m (wheelbase L) - matches vehicle_algo.py
VEHICLE_WIDTH = 2.0   # m
MAX_STEERING_ANGLE = math.radians(30)  # rad

# Simulation timing
SIM_DT = 0.1  # simulation time step (s)
PREDICTION_UPDATE_INTERVAL = 0.2  # how often to run collision detection (s)

# Visualization
SHOW_UNCERTAINTY = True
SHOW_PREDICTIONS = True
SHOW_LABELS = True  # Show vehicle IDs and road names
SHOW_LANE_MARKINGS = True  # Draw lane dividers on roads
SHOW_GHOSTS = True  # Ghost trajectories
SHOW_TRAILS = True  # Fading position trails
SHOW_VELOCITY_ARROWS = True  # Velocity vectors
SHOW_SIGMA_BANDS = True  # Multiple sigma bands (1σ, 2σ, 3σ)
ELLIPSE_SIGMA = 2.0  # draw 2-sigma ellipses
NUM_GHOST_STEPS = 5  # Number of ghost trajectory points
TRAIL_LENGTH = 2.0  # seconds of trail history
TRAIL_MAX_POINTS = 30  # Maximum trail points

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

# ---------- Drawing Helpers ----------
def draw_rounded_rect(surf, rect, color, radius=8, alpha=255):
    """Draw a rounded rectangle with optional transparency"""
    if alpha < 255:
        temp_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, (*color[:3], alpha), temp_surf.get_rect(), border_radius=radius)
        surf.blit(temp_surf, rect.topleft)
    else:
        pygame.draw.rect(surf, color, rect, border_radius=radius)

def draw_uncertainty_ellipse(surf, center_x: int, center_y: int, 
                             P_2x2: np.ndarray, sigma: float, color, scale: float, 
                             line_style='solid', alpha=255):
    """Draw uncertainty ellipse from 2x2 position covariance matrix with various line styles"""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P_2x2)
        if np.any(eigenvalues < 0):
            return
        
        width = 2 * sigma * math.sqrt(max(eigenvalues[0], 0)) * scale
        height = 2 * sigma * math.sqrt(max(eigenvalues[1], 0)) * scale
        angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        n_points = 32
        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            x_local = math.cos(theta) * width / 2
            y_local = math.sin(theta) * height / 2
            x_rot = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_rot = x_local * math.sin(angle) + y_local * math.cos(angle)
            points.append((center_x + x_rot, center_y - y_rot))
        
        if len(points) > 2:
            # Apply alpha if needed
            draw_color = (*color[:3], alpha) if alpha < 255 and len(color) == 3 else color
            
            if line_style == 'solid':
                pygame.draw.polygon(surf, draw_color, points, 1)
            elif line_style == 'dashed':
                # Draw dashed line
                for i in range(0, len(points), 4):
                    if i + 1 < len(points):
                        pygame.draw.line(surf, draw_color, points[i], points[i + 1], 1)
            elif line_style == 'dotted':
                # Draw dotted line
                for i in range(0, len(points), 3):
                    pygame.draw.circle(surf, draw_color, (int(points[i][0]), int(points[i][1])), 1)
    except:
        pass

def draw_sigma_bands(surf, center_x: int, center_y: int, P_2x2: np.ndarray, 
                     color, scale: float, alpha=255):
    """Draw 1σ (solid), 2σ (dashed), 3σ (dotted) uncertainty bands"""
    if SHOW_SIGMA_BANDS:
        draw_uncertainty_ellipse(surf, center_x, center_y, P_2x2, 1.0, color, scale, 'solid', alpha)
        draw_uncertainty_ellipse(surf, center_x, center_y, P_2x2, 2.0, color, scale, 'dashed', alpha)
        draw_uncertainty_ellipse(surf, center_x, center_y, P_2x2, 3.0, color, scale, 'dotted', alpha)
    else:
        draw_uncertainty_ellipse(surf, center_x, center_y, P_2x2, ELLIPSE_SIGMA, color, scale, 'solid', alpha)

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

# ---------- Vehicle Wrapper (uses vehicle_algo.vehicle) ----------
class Vehicle:
    """Wrapper around vehicle_algo.vehicle that handles road-following and rendering"""
    
    def __init__(self, G: RoadGraph, vehicle_id: int):
        self.id = vehicle_id
        self.G = G
        self.color = AGENT_COLORS[vehicle_id % len(AGENT_COLORS)]
        
        # Road-following state
        candidates = [e for e in G.edges if float(e.get("length_m", 0.0)) > 0.1]
        self.edge = random.choice(candidates)
        self.u = self.edge["from"]
        self.v = self.edge["to"]
        self.t = random.random()
        self.prev_u = None
        
        # Lane selection
        self.num_lanes = self._get_num_lanes(self.edge)
        self.lane = random.randint(0, self.num_lanes - 1) if self.num_lanes > 1 else 0
        
        # Initialize position and speed
        lon, lat = G.interp_lonlat(self.u, self.v, self.t)
        x, y = G.lonlat_to_m(lon, lat)
        heading = G.heading_rad(self.u, self.v)
        
        # Apply lane offset
        lane_dx, lane_dy = self._get_lane_offset(heading)
        x += lane_dx
        y += lane_dy
        
        # Varied speed classes
        speed_class = random.random()
        if speed_class < 0.3:
            speed = random.uniform(SPEED_MIN, SPEED_MIN + 3.0)
        elif speed_class < 0.7:
            speed = random.uniform(SPEED_MIN + 3.0, SPEED_MAX - 3.0)
        else:
            speed = random.uniform(SPEED_MAX - 3.0, SPEED_MAX)
        
        self.target_speed = speed
        
        # Create vehicle_algo instance
        # The algo handles UKF internally
        self.algo_vehicle = VehicleAlgo(
            L=VEHICLE_LENGTH,
            external_state=[x, y, speed, heading],
            internal_state=[x, y, speed, heading],
            control_input=[0.0, 0.0]  # [acceleration, steering_angle]
        )
        
        # For collision detection flagging
        self.is_collision_risk = False
        self.collision_probability = 0.0
        
        # Position history for trails
        self.position_history = []  # List of (x, y, timestamp)
        self.sim_time = 0.0
    
    def _get_num_lanes(self, edge) -> int:
        """Parse lane count from edge"""
        num_lanes = edge.get("lanes", "1")
        try:
            if isinstance(num_lanes, list):
                num_lanes = int(num_lanes[0]) if num_lanes else 1
            elif isinstance(num_lanes, str):
                num_lanes = int(num_lanes)
            else:
                num_lanes = int(num_lanes)
        except:
            num_lanes = 1
        return max(1, num_lanes)
    
    def _get_lane_offset(self, heading: float) -> Tuple[float, float]:
        """Calculate lateral offset for current lane"""
        if not USE_LANES or self.num_lanes <= 1:
            return (0.0, 0.0)
        
        lane_offset = (self.lane - (self.num_lanes - 1) / 2.0) * LANE_WIDTH
        perp_angle = heading - math.pi / 2
        dx = lane_offset * math.cos(perp_angle)
        dy = lane_offset * math.sin(perp_angle)
        return (dx, dy)
    
    def _choose_next_edge(self):
        """Choose next edge when reaching intersection"""
        outs = self.G.out.get(self.u, [])
        if not outs:
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
            while d > math.pi: d -= 2*math.pi
            cands.append((abs(d), d, e))
        
        # Weighted selection (prefer straight paths)
        eps = 1e-3
        weights = [max(1.0 / (absd + eps), 0.05) for (absd, _, _) in cands]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for w, (_, _, e) in zip(weights, cands):
            acc += w
            if r <= acc:
                return e
        return cands[0][2]
    
    def _update_lane_for_new_edge(self):
        """Update lane when transitioning edges"""
        self.num_lanes = self._get_num_lanes(self.edge)
        if self.lane >= self.num_lanes:
            self.lane = self.num_lanes - 1
        if random.random() < 0.01 and self.num_lanes > 1:
            self.lane = random.randint(0, self.num_lanes - 1)
    
    def _compute_control_inputs(self) -> Tuple[float, float]:
        """Compute steering and acceleration for road following"""
        # Get current state from algo
        x, y, v, psi = self.algo_vehicle.external_state
        
        # Speed control
        accel = (self.target_speed - v) * 0.5
        accel = np.clip(accel, -3.0, 2.0)
        
        # Steering to follow road
        target_heading = self.G.heading_rad(self.u, self.v)
        heading_error = target_heading - psi
        while heading_error > math.pi: heading_error -= 2*math.pi
        while heading_error < -math.pi: heading_error += 2*math.pi
        
        steering = np.clip(heading_error * 2.0, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        
        return accel, steering
    
    def update(self, dt: float):
        """Update vehicle state"""
        # Update simulation time
        self.sim_time += dt
        
        # Compute control inputs
        accel, steering = self._compute_control_inputs()
        
        # Update the algo vehicle (this handles UKF internally)
        self.algo_vehicle.update_external([accel, steering], dt)
        
        # Update road position
        x, y, v, psi = self.algo_vehicle.external_state
        
        # Track position history for trails
        if SHOW_TRAILS:
            self.position_history.append((x, y, self.sim_time))
            # Keep only recent history
            cutoff_time = self.sim_time - TRAIL_LENGTH
            self.position_history = [(px, py, t) for px, py, t in self.position_history if t >= cutoff_time]
            # Limit max points
            if len(self.position_history) > TRAIL_MAX_POINTS:
                self.position_history = self.position_history[-TRAIL_MAX_POINTS:]
        
        L = max(float(self.edge.get("length_m", 1.0)), 1e-6)
        self.t += (v * dt) / L
        
        # Handle edge transitions
        if self.t >= 1.0:
            self.t = 0.0
            self.prev_u = self.u
            self.u = self.v
            self.edge = self._choose_next_edge()
            self.v = self.edge["to"]
            self._update_lane_for_new_edge()
            
            # Occasionally change target speed
            if random.random() < 0.3:
                speed_class = random.random()
                if speed_class < 0.3:
                    self.target_speed = random.uniform(SPEED_MIN, SPEED_MIN + 3.0)
                elif speed_class < 0.7:
                    self.target_speed = random.uniform(SPEED_MIN + 3.0, SPEED_MAX - 3.0)
                else:
                    self.target_speed = random.uniform(SPEED_MAX - 3.0, SPEED_MAX)
        
        # Constrain to road with lane offset
        lon, lat = self.G.interp_lonlat(self.u, self.v, self.t)
        road_x, road_y = self.G.lonlat_to_m(lon, lat)
        road_heading = self.G.heading_rad(self.u, self.v)
        
        lane_dx, lane_dy = self._get_lane_offset(road_heading)
        road_x += lane_dx
        road_y += lane_dy
        
        # Blend road position with physics model
        blend = 0.8
        self.algo_vehicle.external_state[0] = blend * road_x + (1 - blend) * self.algo_vehicle.external_state[0]
        self.algo_vehicle.external_state[1] = blend * road_y + (1 - blend) * self.algo_vehicle.external_state[1]
        self.algo_vehicle.external_state[3] = blend * road_heading + (1 - blend) * self.algo_vehicle.external_state[3]
    
    def m_to_lonlat(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert meters to lon/lat"""
        lon = x_m / self.G.m_per_deg_lon
        lat = y_m / self.G.m_per_deg_lat
        return lon, lat
    
    def draw(self, surf, world_to_screen, scale, pan, base_scale, font, time_pulse):
        """Draw vehicle with all visual enhancements"""
        x, y, v, psi = self.algo_vehicle.external_state
        lon, lat = self.m_to_lonlat(x, y)
        px, py = world_to_screen(lon, lat, scale=scale, pan=pan)
        
        # Calculate proper scale from meters to pixels
        # base_scale is pixels per degree, we need pixels per meter
        meters_to_pixels = base_scale / self.G.m_per_deg_lon * scale
        
        # Determine color based on risk level
        is_high_risk = self.is_collision_risk
        if is_high_risk:
            if self.collision_probability > 0.5:
                color = COLOR_DANGER
            else:
                color = COLOR_CAUTION
        else:
            color = self.color
        
        # Draw fading trail
        if SHOW_TRAILS and len(self.position_history) > 1:
            for i in range(len(self.position_history) - 1):
                px1, py1, t1 = self.position_history[i]
                px2, py2, t2 = self.position_history[i + 1]
                
                # Convert to screen coordinates
                lon1, lat1 = self.m_to_lonlat(px1, py1)
                lon2, lat2 = self.m_to_lonlat(px2, py2)
                spx1, spy1 = world_to_screen(lon1, lat1, scale=scale, pan=pan)
                spx2, spy2 = world_to_screen(lon2, lat2, scale=scale, pan=pan)
                
                # Exponential alpha decay based on age
                age = self.sim_time - t1
                alpha = int(200 * math.exp(-age / (TRAIL_LENGTH * 0.4)))
                alpha = max(20, min(200, alpha))
                
                # Draw trail segment with alpha
                trail_surf = pygame.Surface((abs(spx2 - spx1) + 4, abs(spy2 - spy1) + 4), pygame.SRCALPHA)
                trail_color = (*color[:3], alpha)
                local_p1 = (2, 2) if spx1 <= spx2 else (abs(spx2 - spx1) + 2, 2)
                local_p2 = (abs(spx2 - spx1) + 2, abs(spy2 - spy1) + 2) if spx2 >= spx1 else (2, abs(spy2 - spy1) + 2)
                if spy1 > spy2:
                    local_p1 = (local_p1[0], abs(spy2 - spy1) + 2)
                    local_p2 = (local_p2[0], 2)
                pygame.draw.line(trail_surf, trail_color, local_p1, local_p2, 2)
                surf.blit(trail_surf, (min(spx1, spx2) - 2, min(spy1, spy2) - 2))
        
        # Draw ghost trajectories (future positions)
        if SHOW_GHOSTS:
            ghost_dt = 0.3  # Time step between ghost positions
            for i in range(1, NUM_GHOST_STEPS + 1):
                # Simple forward projection (could be improved with actual prediction)
                ghost_x = x + v * math.cos(psi) * ghost_dt * i
                ghost_y = y + v * math.sin(psi) * ghost_dt * i
                ghost_lon, ghost_lat = self.m_to_lonlat(ghost_x, ghost_y)
                ghost_px, ghost_py = world_to_screen(ghost_lon, ghost_lat, scale=scale, pan=pan)
                
                # Fading alpha for ghosts
                alpha = int(150 * (1.0 - i / (NUM_GHOST_STEPS + 1)))
                ghost_color = (*color[:3], alpha)
                
                # Draw small circle for ghost position
                ghost_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(ghost_surf, ghost_color, (5, 5), 3)
                surf.blit(ghost_surf, (ghost_px - 5, ghost_py - 5))
        
        # Draw velocity arrow
        if SHOW_VELOCITY_ARROWS:
            arrow_len = min(40, v * 3)  # Scale with velocity
            arrow_end_x = px + arrow_len * math.cos(psi)
            arrow_end_y = py - arrow_len * math.sin(psi)  # Negative because screen Y is inverted
            
            # Draw arrow with alpha
            vel_surf = pygame.Surface((abs(int(arrow_end_x - px)) + 10, abs(int(arrow_end_y - py)) + 10), pygame.SRCALPHA)
            vel_color = (*color[:3], 180)
            start_local = (5, 5)
            end_local = (int(arrow_end_x - px) + 5, int(arrow_end_y - py) + 5)
            pygame.draw.line(vel_surf, vel_color, start_local, end_local, 2)
            # Arrow head
            arrow_angle = psi
            head_len = 6
            left_x = arrow_end_x - head_len * math.cos(arrow_angle - math.radians(25))
            left_y = arrow_end_y + head_len * math.sin(arrow_angle - math.radians(25))
            right_x = arrow_end_x - head_len * math.cos(arrow_angle + math.radians(25))
            right_y = arrow_end_y + head_len * math.sin(arrow_angle + math.radians(25))
            pygame.draw.line(vel_surf, vel_color, end_local, (int(left_x - px) + 5, int(left_y - py) + 5), 2)
            pygame.draw.line(vel_surf, vel_color, end_local, (int(right_x - px) + 5, int(right_y - py) + 5), 2)
            surf.blit(vel_surf, (min(px, int(arrow_end_x)) - 5, min(py, int(arrow_end_y)) - 5))
        
        # Draw vehicle as arrow
        tip = (px, py)
        tail = (px - ARROW_LEN * math.cos(psi), py + ARROW_LEN * math.sin(psi))
        left = (tip[0] - ARROW_LEN * math.cos(psi - ARROW_ANG),
                tip[1] + ARROW_LEN * math.sin(psi - ARROW_ANG))
        right = (tip[0] - ARROW_LEN * math.cos(psi + ARROW_ANG),
                 tip[1] + ARROW_LEN * math.sin(psi + ARROW_ANG))
        
        # Outline with glow for high risk
        outline_width = 7 if is_high_risk else 5
        outline_color = (255, 100, 80) if is_high_risk else (40, 40, 50)
        pygame.draw.line(surf, outline_color, tail, tip, outline_width)
        pygame.draw.line(surf, outline_color, tip, left, outline_width)
        pygame.draw.line(surf, outline_color, tip, right, outline_width)
        
        # Body
        line_width = 5 if is_high_risk else 3
        pygame.draw.line(surf, color, tail, tip, line_width)
        pygame.draw.line(surf, color, tip, left, line_width)
        pygame.draw.line(surf, color, tip, right, line_width)
        
        # Improved label with rounded dark chip
        if SHOW_LABELS:
            speed_kmh = v * 3.6
            label_text = font.render(f"V{self.id} {speed_kmh:.0f}km/h", True, (240, 240, 245))
            label_rect = label_text.get_rect(center=(px, py - 26))
            
            # Dark rounded background
            bg_rect = label_rect.inflate(12, 6)
            draw_rounded_rect(surf, bg_rect, (30, 35, 45), radius=6, alpha=220)
            surf.blit(label_text, label_rect)
            
            # Lane indicator (if applicable)
            if USE_LANES and self.num_lanes > 1:
                lane_color = (120, 180, 220)
                lane_indicator_y = py - 42
                for i in range(self.num_lanes):
                    dot_x = px - (self.num_lanes - 1) * 3 + i * 6
                    if i == self.lane:
                        pygame.draw.circle(surf, lane_color, (dot_x, lane_indicator_y), 3)
                    else:
                        pygame.draw.circle(surf, (100, 100, 100), (dot_x, lane_indicator_y), 2)
        
        # Uncertainty ellipse with breathing animation
        if SHOW_UNCERTAINTY:
            P_2x2 = self.algo_vehicle.P[:2, :2]
            # Breathing effect: pulse alpha between 180-255
            breath_alpha = int(220 + 35 * math.sin(time_pulse * 2 * math.pi))
            draw_sigma_bands(surf, px, py, P_2x2, color, meters_to_pixels, breath_alpha)
        
        # Draw predictions
        if SHOW_PREDICTIONS:
            pred_x = self.algo_vehicle.pred_next_state[0]
            pred_y = self.algo_vehicle.pred_next_state[1]
            if pred_x != 0 or pred_y != 0:
                pred_lon, pred_lat = self.m_to_lonlat(pred_x, pred_y)
                pred_px, pred_py = world_to_screen(pred_lon, pred_lat, scale=scale, pan=pan)
                pygame.draw.circle(surf, color, (pred_px, pred_py), 5, 2)
                
                # Draw prediction uncertainty
                P_pred_2x2 = self.algo_vehicle.predicted_P[:2, :2]
                draw_sigma_bands(surf, pred_px, pred_py, P_pred_2x2, 
                               tuple(int(c * 0.7) for c in color[:3]), meters_to_pixels, 150)

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
def check_collisions(vehicles: List[Vehicle]):
    """Use vehicle_algo's celestial mechanics b-plane collision detection"""
    # Reset collision flags
    for v in vehicles:
        v.is_collision_risk = False
        v.collision_probability = 0.0
    
    # Make sure all vehicles have predictions (ukf_predict needs to be called)
    for v in vehicles:
        v.algo_vehicle.ukf_predict(SIM_DT)
    
    # Use vehicle_simulation to check for collisions
    algo_vehicles = [v.algo_vehicle for v in vehicles]
    sim = vehicle_simulation(algo_vehicles)
    
    # Check collision at 95% confidence level
    is_collision, idx1, idx2 = sim.is_collision(0.99)
    
    # Calculate actual Pc for the pair (simplified - would need actual b-plane calculation)
    collision_prob = 0.0
    if is_collision and idx1 is not None and idx2 is not None:
        # Estimate collision probability based on distance and uncertainty
        v1_pos = vehicles[idx1].algo_vehicle.external_state[:2]
        v2_pos = vehicles[idx2].algo_vehicle.external_state[:2]
        dist = np.linalg.norm(np.array(v1_pos) - np.array(v2_pos))
        # Simple heuristic: closer = higher probability
        collision_prob = max(0.01, min(0.99, math.exp(-dist / 20.0)))
        
        vehicles[idx1].is_collision_risk = True
        vehicles[idx2].is_collision_risk = True
        vehicles[idx1].collision_probability = collision_prob
        vehicles[idx2].collision_probability = collision_prob
        return True, idx1, idx2, collision_prob
    
    return False, None, None, 0.0

def draw_bplane_panel(screen, vehicles, collision_pair, collision_prob, font, time):
    """Draw b-plane visualization panel when collision pair exists"""
    if collision_pair[0] is None or collision_pair[1] is None:
        return
    
    v1 = vehicles[collision_pair[0]]
    v2 = vehicles[collision_pair[1]]
    
    # Panel dimensions and position (bottom-right)
    panel_width = 280
    panel_height = 220
    panel_x = WIN_W - panel_width - 15
    panel_y = WIN_H - panel_height - 15
    
    # Draw panel background
    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    pygame.draw.rect(panel_surf, (30, 35, 45, 240), panel_surf.get_rect(), border_radius=10)
    pygame.draw.rect(panel_surf, (80, 90, 110, 200), panel_surf.get_rect(), width=2, border_radius=10)
    screen.blit(panel_surf, (panel_x, panel_y))
    
    # Title
    title_font = pygame.font.SysFont("Arial", 13, bold=True)
    title = title_font.render("B-Plane Collision Analysis", True, (220, 230, 240))
    screen.blit(title, (panel_x + 15, panel_y + 12))
    
    # Vehicle pair info
    pair_text = font.render(f"V{collision_pair[0]} ↔ V{collision_pair[1]}", True, (200, 210, 220))
    screen.blit(pair_text, (panel_x + 15, panel_y + 35))
    
    # Large Pc display
    pc_font = pygame.font.SysFont("Arial", 32, bold=True)
    pc_color = COLOR_DANGER if collision_prob > 0.5 else COLOR_CAUTION if collision_prob > 0.1 else COLOR_SAFE
    pc_text = pc_font.render(f"{collision_prob:.3f}", True, pc_color)
    pc_label = font.render("Pc:", True, (180, 190, 200))
    screen.blit(pc_label, (panel_x + 15, panel_y + 60))
    screen.blit(pc_text, (panel_x + 50, panel_y + 52))
    
    # Draw mini b-plane visualization (simplified)
    bplane_center_x = panel_x + panel_width // 2
    bplane_center_y = panel_y + 140
    bplane_radius = 45
    
    # Draw coordinate axes
    pygame.draw.line(screen, (100, 110, 120), 
                    (bplane_center_x - bplane_radius - 5, bplane_center_y),
                    (bplane_center_x + bplane_radius + 5, bplane_center_y), 1)
    pygame.draw.line(screen, (100, 110, 120),
                    (bplane_center_x, bplane_center_y - bplane_radius - 5),
                    (bplane_center_x, bplane_center_y + bplane_radius + 5), 1)
    
    # Draw collision circle (represents combined vehicle radii)
    collision_radius = int(bplane_radius * 0.25)
    pygame.draw.circle(screen, COLOR_DANGER, (bplane_center_x, bplane_center_y), 
                      collision_radius, 2)
    
    # Draw uncertainty ellipse (simplified 2D projection)
    v1_pos = v1.algo_vehicle.external_state[:2]
    v2_pos = v2.algo_vehicle.external_state[:2]
    rel_pos = np.array(v2_pos) - np.array(v1_pos)
    dist = np.linalg.norm(rel_pos)
    
    if dist > 0:
        # Normalize and scale for display
        display_offset = (rel_pos / dist) * min(bplane_radius * 0.7, dist * 0.5)
        miss_x = int(bplane_center_x + display_offset[0])
        miss_y = int(bplane_center_y - display_offset[1])  # Invert Y for screen coords
        
        # Draw miss point
        pygame.draw.circle(screen, COLOR_SAFE, (miss_x, miss_y), 4)
        pygame.draw.circle(screen, (240, 240, 245), (miss_x, miss_y), 2)
        
        # Draw uncertainty ellipse around miss point
        # Simplified: draw circle proportional to combined uncertainty
        P1 = v1.algo_vehicle.P[:2, :2]
        P2 = v2.algo_vehicle.P[:2, :2]
        combined_uncertainty = np.trace(P1 + P2) ** 0.5
        unc_radius = int(min(30, combined_uncertainty * 2))
        pygame.draw.circle(screen, (*COLOR_CAUTION, 120), (miss_x, miss_y), unc_radius, 1)
    
    # Labels
    axis_font = pygame.font.SysFont("Arial", 9)
    x_label = axis_font.render("ξ", True, (140, 150, 160))
    y_label = axis_font.render("ζ", True, (140, 150, 160))
    screen.blit(x_label, (bplane_center_x + bplane_radius + 8, bplane_center_y - 5))
    screen.blit(y_label, (bplane_center_x - 5, bplane_center_y - bplane_radius - 12))
    
    # Distance info
    dist_m = np.linalg.norm(np.array(v1_pos) - np.array(v2_pos))
    dist_text = font.render(f"Distance: {dist_m:.1f}m", True, (180, 190, 200))
    screen.blit(dist_text, (panel_x + 15, panel_y + 195))

def draw_collision_badge(screen, vehicles, collision_pair, collision_prob, w2s, scale, pan, font):
    """Draw on-map collision badge showing Pc"""
    if collision_pair[0] is None or collision_pair[1] is None:
        return
    
    v1 = vehicles[collision_pair[0]]
    v2 = vehicles[collision_pair[1]]
    
    # Calculate midpoint between vehicles
    v1_pos = v1.algo_vehicle.external_state[:2]
    v2_pos = v2.algo_vehicle.external_state[:2]
    mid_x = (v1_pos[0] + v2_pos[0]) / 2
    mid_y = (v1_pos[1] + v2_pos[1]) / 2
    
    mid_lon, mid_lat = v1.m_to_lonlat(mid_x, mid_y)
    badge_px, badge_py = w2s(mid_lon, mid_lat, scale=scale, pan=pan)
    
    # Draw badge
    badge_text = f"Pc {collision_prob:.3f}"
    badge_color = COLOR_DANGER if collision_prob > 0.5 else COLOR_CAUTION
    text_render = font.render(badge_text, True, (255, 255, 255))
    text_rect = text_render.get_rect(center=(badge_px, badge_py))
    
    # Background
    bg_rect = text_rect.inflate(10, 6)
    draw_rounded_rect(screen, bg_rect, badge_color, radius=6, alpha=230)
    screen.blit(text_render, text_rect)
    
    # Draw connecting lines to both vehicles
    v1_lon, v1_lat = v1.m_to_lonlat(v1_pos[0], v1_pos[1])
    v2_lon, v2_lat = v2.m_to_lonlat(v2_pos[0], v2_pos[1])
    v1_px, v1_py = w2s(v1_lon, v1_lat, scale=scale, pan=pan)
    v2_px, v2_py = w2s(v2_lon, v2_lat, scale=scale, pan=pan)
    
    line_color = (*badge_color, 150)
    line_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.line(line_surf, line_color, (v1_px, v1_py), (badge_px, badge_py), 1)
    pygame.draw.line(line_surf, line_color, (v2_px, v2_py), (badge_px, badge_py), 1)
    screen.blit(line_surf, (0, 0))

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
    pygame.display.set_caption("B-Plane Collision Prediction — Space: pause | G: ghosts | T: trails | V: velocity | S: sigma bands")
    clock = pygame.time.Clock()

    w2s, bounds, base_scale, (base_offx, base_offy) = make_projector(nodes, WIN_W, WIN_H, MARGIN)

    scale = 1.0
    pan = [0.0, 0.0]
    dragging = False
    last_mouse = (0, 0)
    show_buildings = len(buildings) > 0  # Show buildings by default if available

    G = RoadGraph(nodes, edges)
    vehicles = [Vehicle(G, i) for i in range(NUM_AGENTS)]
    
    # Collision check timer
    collision_timer = 0.0
    collision_detected = False
    collision_pair = (None, None)
    collision_prob = 0.0
    
    # Animation timing
    sim_time = 0.0
    paused = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if not paused:
            sim_time += dt

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
                elif ev.key == pygame.K_g:
                    global SHOW_GHOSTS
                    SHOW_GHOSTS = not SHOW_GHOSTS
                elif ev.key == pygame.K_t:
                    global SHOW_TRAILS
                    SHOW_TRAILS = not SHOW_TRAILS
                elif ev.key == pygame.K_v:
                    global SHOW_VELOCITY_ARROWS
                    SHOW_VELOCITY_ARROWS = not SHOW_VELOCITY_ARROWS
                elif ev.key == pygame.K_s:
                    global SHOW_SIGMA_BANDS
                    SHOW_SIGMA_BANDS = not SHOW_SIGMA_BANDS
                elif ev.key == pygame.K_SPACE:
                    paused = not paused

        # Update vehicles (only when not paused)
        if not paused:
            for v in vehicles:
                v.update(SIM_DT)

            # Check for collisions periodically
            collision_timer += dt
            if collision_timer >= PREDICTION_UPDATE_INTERVAL:
                collision_timer = 0.0
                collision_detected, idx1, idx2, cprob = check_collisions(vehicles)
                if collision_detected:
                    collision_pair = (idx1, idx2)
                    collision_prob = cprob
                else:
                    collision_pair = (None, None)
                    collision_prob = 0.0
        
        # Calculate time pulse for breathing animation (0 to 1)
        time_pulse = (sim_time % 3.0) / 3.0

        # Draw
        screen.fill(BG)
        
        # Draw buildings first (background layer)
        if show_buildings and buildings:
            draw_buildings(screen, buildings, w2s, scale, pan)
        
        # Draw roads on top of buildings
        font_small = pygame.font.SysFont("Arial", 11, bold=True)
        draw_edges(screen, nodes, edges, w2s, scale, pan, base_scale, scale, font_small)
        
        # Draw on-map collision badge
        if collision_detected and collision_pair[0] is not None:
            draw_collision_badge(screen, vehicles, collision_pair, collision_prob, w2s, scale, pan, font_small)
        
        # Draw vehicles on top
        for v in vehicles:
            v.draw(screen, w2s, scale, pan, base_scale, font_small, time_pulse)

        # ========== Professional UI Panel ==========
        panel_font = pygame.font.SysFont("Arial", 13)
        title_font = pygame.font.SysFont("Arial", 15, bold=True)
        
        # Top control panel (expanded for extra line)
        panel_height = 85
        panel_surf = pygame.Surface((WIN_W, panel_height))
        panel_surf.fill((30, 35, 45))
        panel_surf.set_alpha(240)
        screen.blit(panel_surf, (0, 0))
        
        # Title
        title = title_font.render("Vehicle Collision Prediction Simulation", True, (200, 220, 240))
        screen.blit(title, (15, 10))
        
        # Controls (split into two lines for readability)
        controls1 = panel_font.render("Controls: Right-drag=Pan | Wheel=Zoom | R=Reset | Space=Pause", True, (160, 175, 195))
        controls2 = panel_font.render("Toggles: G=Ghosts | T=Trails | V=VelArrows | S=SigmaBands | U=Uncertainty | P=Pred | B=Buildings | L=Labels", True, (160, 175, 195))
        screen.blit(controls1, (15, 32))
        screen.blit(controls2, (15, 48))
        
        # Status indicators
        status_items = []
        status_items.append(f"Vehicles: {NUM_AGENTS}")
        status_items.append(f"{'⏸' if paused else '▶'} {'PAUSED' if paused else 'Running'}")
        status_items.append(f"Ghosts: {'✓' if SHOW_GHOSTS else '✗'}")
        status_items.append(f"Trails: {'✓' if SHOW_TRAILS else '✗'}")
        status_items.append(f"VelArrows: {'✓' if SHOW_VELOCITY_ARROWS else '✗'}")
        status_items.append(f"Σ-Bands: {'✓' if SHOW_SIGMA_BANDS else '✗'}")
        status_items.append(f"FPS: {int(clock.get_fps())}")
        
        status_text = " | ".join(status_items)
        status_render = panel_font.render(status_text, True, (140, 160, 180))
        screen.blit(status_render, (15, 65))
        
        # B-plane visualization panel (replaces old collision warning)
        if collision_detected and collision_pair[0] is not None:
            draw_bplane_panel(screen, vehicles, collision_pair, collision_prob, panel_font, sim_time)
        
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
