# collision_prediction_sim.py
# Vehicle simulation demonstrating celestial mechanics b-plane collision prediction
# Uses vehicle_algo.py for UKF state estimation and collision detection

import json, math, random, sys
from pathlib import Path
import pygame
import numpy as np
from typing import List, Tuple
from utils.vehicle_algo import vehicle as VehicleAlgo, vehicle_simulation
from utils.crash_tracker import CrashTracker
from utils.detailed_crash_tracker import DetailedCrashTracker

# ---------- Display / Style ----------
WIN_W, WIN_H = 1200, 800
MARGIN = 60
BG = (235, 238, 242)  # Soft blue-gray background
DEFAULT_WIDTH, DEFAULT_COLOR = 3, (255, 255, 255)

# Zoom limits for better UX
MIN_ZOOM = 0.3
MAX_ZOOM = 5.0

# Road hierarchy for zoom-based filtering
ROAD_HIERARCHY = {
    "motorway": 1,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "residential": 5,
    "unclassified": 5,
    "service": 6,
    "living_street": 6,
    "footway": 7,
    "path": 7
}

# Realistic road styling with refined colors and thinner roads
HIGHWAY_STYLE = {
    "motorway": {
        "width": 8, 
        "color": (255, 180, 80),  # Warm orange
        "outline": (230, 140, 50),
        "lanes": 4,
        "label_color": (100, 70, 20),
        "priority": 1
    },
    "trunk": {
        "width": 7, 
        "color": (255, 195, 100),
        "outline": (230, 150, 60),
        "lanes": 4,
        "label_color": (100, 75, 25),
        "priority": 1
    },
    "primary": {
        "width": 6, 
        "color": (255, 220, 120),  # Softer yellow
        "outline": (220, 180, 85),
        "lanes": 3,
        "label_color": (100, 85, 40),
        "priority": 2
    },
    "secondary": {
        "width": 5, 
        "color": (255, 240, 160),  # Pale yellow
        "outline": (210, 200, 130),
        "lanes": 2,
        "label_color": (100, 100, 60),
        "priority": 3
    },
    "tertiary": {
        "width": 4, 
        "color": (255, 255, 255),  # White
        "outline": (190, 190, 190),
        "lanes": 2,
        "label_color": (80, 80, 80),
        "priority": 4
    },
    "residential": {
        "width": 4, 
        "color": (255, 255, 255),  # White
        "outline": (200, 200, 200),
        "lanes": 2,
        "label_color": (100, 100, 100),
        "priority": 5
    },
    "unclassified": {
        "width": 3, 
        "color": (252, 252, 252),
        "outline": (210, 210, 210),
        "lanes": 2,
        "label_color": (120, 120, 120),
        "priority": 5
    },
    "service": {
        "width": 3, 
        "color": (248, 248, 248),
        "outline": (220, 220, 220),
        "lanes": 1,
        "label_color": (140, 140, 140),
        "priority": 6
    },
    "living_street": {
        "width": 3, 
        "color": (250, 250, 250),
        "outline": (215, 215, 215),
        "lanes": 1,
        "label_color": (130, 130, 130),
        "priority": 6
    },
    "footway": {
        "width": 2, 
        "color": (230, 230, 230),
        "outline": (200, 200, 200),
        "lanes": 0,
        "label_color": (150, 150, 150),
        "priority": 7
    },
    "path": {
        "width": 2, 
        "color": (225, 225, 225),
        "outline": (195, 195, 195),
        "lanes": 0,
        "label_color": (150, 150, 150),
        "priority": 7
    },
}
ARROW_LEN = 14  # px
ARROW_ANG = math.radians(22)

# Camera control settings
PAN_SPEED = 15.0  # pixels per frame for keyboard panning
ZOOM_SPEED = 0.05  # zoom increment per frame

# ---------- Simulation Parameters ----------
NUM_AGENTS = 60
SPEED_MIN = 3.0   # m/s (slower vehicles)
SPEED_MAX = 20.0  # m/s (faster vehicles - more variety)

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
        "label_color": (100, 100, 100),
        "priority": 5
    }
    return HIGHWAY_STYLE.get(hwy, default)

def should_draw_road(highway_type: str, zoom_level: float) -> bool:
    """Determine if a road should be drawn at the current zoom level"""
    hierarchy = ROAD_HIERARCHY.get(highway_type, 5)
    
    # Zoom thresholds for different road types
    if zoom_level < 0.5:
        return hierarchy <= 1  # Only motorways/trunks
    elif zoom_level < 0.8:
        return hierarchy <= 2  # Add primary roads
    elif zoom_level < 1.2:
        return hierarchy <= 3  # Add secondary roads
    elif zoom_level < 1.8:
        return hierarchy <= 4  # Add tertiary roads
    elif zoom_level < 2.5:
        return hierarchy <= 5  # Add residential/unclassified
    else:
        return True  # Show all roads when zoomed in

def get_adaptive_width(base_width: int, zoom_level: float) -> int:
    """Scale road width based on zoom level for better visibility"""
    # At low zoom, make roads slightly thicker for visibility
    # At high zoom, use normal width
    if zoom_level < 0.5:
        return int(base_width * 1.4)
    elif zoom_level < 0.8:
        return int(base_width * 1.2)
    elif zoom_level < 1.2:
        return int(base_width * 1.05)
    else:
        return base_width

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
def draw_buildings(screen, buildings, w2s, scale, pan, zoom_level):
    """Draw building footprints as realistic structures (only when zoomed in)"""
    # Only show buildings when zoomed in enough
    if zoom_level < 1.3:
        return
    
    building_fill = (210, 215, 225)  # Subtle blue-gray
    building_outline = (180, 185, 195)  # Soft outline
    
    buildings_drawn = 0
    max_buildings = 500  # Limit for performance
    
    for building in buildings:
        if buildings_drawn >= max_buildings:
            break
            
        if building['type'] == 'Polygon':
            coords = building['coordinates']
            if len(coords) < 3:
                continue
            
            # Convert to screen coordinates
            screen_coords = []
            on_screen = False
            for lon, lat in coords:
                px, py = w2s(lon, lat, scale=scale, pan=pan)
                screen_coords.append((px, py))
                # Check if at least one point is on screen
                if 0 <= px <= WIN_W and 0 <= py <= WIN_H:
                    on_screen = True
            
            # Only draw if visible on screen
            if on_screen and len(screen_coords) >= 3:
                try:
                    # Draw filled building
                    pygame.draw.polygon(screen, building_fill, screen_coords, 0)
                    # Draw outline
                    pygame.draw.polygon(screen, building_outline, screen_coords, 2)
                    buildings_drawn += 1
                except:
                    pass  # Skip invalid polygons

def draw_minimap(screen, nodes, edges, vehicles, bounds, pan, scale):
    """Draw a minimap in the top-right corner"""
    minimap_size = 150
    minimap_x = WIN_W - minimap_size - 15
    minimap_y = 100  # Below the control panel
    
    # Draw minimap background
    minimap_surf = pygame.Surface((minimap_size, minimap_size), pygame.SRCALPHA)
    pygame.draw.rect(minimap_surf, (30, 35, 45, 220), minimap_surf.get_rect(), border_radius=8)
    pygame.draw.rect(minimap_surf, (80, 90, 110, 200), minimap_surf.get_rect(), width=2, border_radius=8)
    screen.blit(minimap_surf, (minimap_x, minimap_y))
    
    # Calculate minimap projection
    min_lon, max_lon, min_lat, max_lat = bounds
    
    def minimap_project(lon, lat):
        # Project to minimap coordinates
        if abs(max_lon - min_lon) < 1e-12 or abs(max_lat - min_lat) < 1e-12:
            return minimap_x + minimap_size // 2, minimap_y + minimap_size // 2
        
        x = minimap_x + 10 + (lon - min_lon) / (max_lon - min_lon) * (minimap_size - 20)
        y = minimap_y + 10 + (max_lat - lat) / (max_lat - min_lat) * (minimap_size - 20)
        return int(x), int(y)
    
    # Draw major roads on minimap
    for e in edges:
        highway_type = e.get("highway", "unclassified")
        hierarchy = ROAD_HIERARCHY.get(highway_type, 5)
        
        # Only show major roads on minimap
        if hierarchy <= 2:
            u, v = e["from"], e["to"]
            if u in nodes and v in nodes:
                a, b = nodes[u], nodes[v]
                p0 = minimap_project(a["x"], a["y"])
                p1 = minimap_project(b["x"], b["y"])
                
                road_color = (255, 180, 80) if hierarchy == 1 else (255, 220, 120)
                pygame.draw.line(screen, road_color, p0, p1, 2 if hierarchy == 1 else 1)
    
    # Draw vehicles as dots
    for v in vehicles:
        x, y, _, _ = v.algo_vehicle.external_state
        lon, lat = v.m_to_lonlat(x, y)
        px, py = minimap_project(lon, lat)
        
        # Draw vehicle dot
        vehicle_color = COLOR_DANGER if v.is_collision_risk else COLOR_SAFE
        pygame.draw.circle(screen, vehicle_color, (px, py), 3)
    
    # Draw viewport indicator (shows current view on minimap)
    # This is a simplified representation
    viewport_color = (120, 180, 220, 150)
    viewport_size = int((minimap_size - 20) / scale)
    viewport_size = max(5, min(viewport_size, minimap_size - 20))
    
    # Calculate viewport center on minimap
    center_lon = min_lon + (max_lon - min_lon) * 0.5
    center_lat = min_lat + (max_lat - min_lat) * 0.5
    vcx, vcy = minimap_project(center_lon, center_lat)
    
    # Draw viewport rectangle
    viewport_rect = pygame.Rect(
        vcx - viewport_size // 2,
        vcy - viewport_size // 2,
        viewport_size,
        viewport_size
    )
    viewport_surf = pygame.Surface((viewport_size, viewport_size), pygame.SRCALPHA)
    pygame.draw.rect(viewport_surf, viewport_color, viewport_surf.get_rect(), width=2)
    screen.blit(viewport_surf, viewport_rect.topleft)

def draw_edges(screen, nodes, edges, w2s, scale, pan, base_scale, zoom_scale, font):
    """Draw roads with realistic styling, lane markings, and labels"""
    drawn_labels = {}  # Track which labels we've drawn to avoid duplicates
    label_count = 0
    max_labels = 50 if zoom_scale > 1.5 else 30 if zoom_scale > 1.0 else 15
    
    # Filter and sort edges by priority (lower priority = draw first)
    edges_to_draw = []
    for e in edges:
        highway_type = e.get("highway", "unclassified")
        if should_draw_road(highway_type, zoom_scale):
            u, v = e["from"], e["to"]
            if u in nodes and v in nodes:
                style = highway_style(highway_type)
                edges_to_draw.append((style["priority"], e, style))
    
    # Sort by priority (higher priority roads drawn last = on top)
    edges_to_draw.sort(key=lambda x: x[0], reverse=True)
    
    for priority, e, style in edges_to_draw:
        u, v = e["from"], e["to"]
        a, b = nodes[u], nodes[v]
        p0 = w2s(a["x"], a["y"], scale=zoom_scale, pan=pan)
        p1 = w2s(b["x"], b["y"], scale=zoom_scale, pan=pan)
        
        # Skip if edge is completely off-screen (basic culling)
        if (max(p0[0], p1[0]) < 0 or min(p0[0], p1[0]) > WIN_W or
            max(p0[1], p1[1]) < 0 or min(p0[1], p1[1]) > WIN_H):
            continue
        
        # Get adaptive width
        base_width = style["width"]
        width = get_adaptive_width(base_width, zoom_scale)
        color = style["color"]
        outline = style["outline"]
        lanes = style["lanes"]
        
        # Draw road outline (subtle border)
        outline_width = max(1, width + 2)
        pygame.draw.line(screen, outline, p0, p1, outline_width)
        
        # Draw road surface
        pygame.draw.line(screen, color, p0, p1, width)
        
        # Draw lane markings for multi-lane roads (only when zoomed in enough)
        if SHOW_LANE_MARKINGS and lanes >= 2 and width >= 6 and zoom_scale > 1.5:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Draw center line (subtle dashed)
                num_dashes = int(length / 15)
                dash_color = (210, 210, 150, 180)  # Subtle yellow
                for i in range(num_dashes):
                    t1 = i / num_dashes
                    t2 = (i + 0.4) / num_dashes
                    dash_start = (int(p0[0] + dx * t1), int(p0[1] + dy * t1))
                    dash_end = (int(p0[0] + dx * t2), int(p0[1] + dy * t2))
                    pygame.draw.line(screen, dash_color[:3], dash_start, dash_end, 1)
        
        # Draw road name labels (only when zoomed in and for major roads)
        if SHOW_LABELS and e.get("name") and label_count < max_labels and zoom_scale > 1.0:
            # Prioritize major roads for labels
            if priority <= 3 or zoom_scale > 1.8:
                road_name = e["name"]
                if isinstance(road_name, list):
                    road_name = road_name[0] if road_name else None
                
                if road_name and isinstance(road_name, str):
                    if road_name not in drawn_labels:
                        mx = (p0[0] + p1[0]) // 2
                        my = (p0[1] + p1[1]) // 2
                        
                        # Only draw if label position is on screen
                        if 0 <= mx <= WIN_W and 0 <= my <= WIN_H:
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
                            label_count += 1

# ---------- Collision Detection ----------
def check_collisions(vehicles: List[Vehicle], crash_tracker: CrashTracker = None, 
                     detailed_tracker: DetailedCrashTracker = None, sim_time: float = 0.0):
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
    
    # Record detection if crash tracker is provided
    if crash_tracker is not None and is_collision and idx1 is not None and idx2 is not None:
        crash_tracker.record_detection(sim_time, idx1, idx2, 0.99)
    
    # Record detailed detection if detailed tracker is provided
    if detailed_tracker is not None and is_collision and idx1 is not None and idx2 is not None:
        # Add id to vehicles for tracking
        vehicles[idx1].algo_vehicle.id = idx1
        vehicles[idx2].algo_vehicle.id = idx2
        detailed_tracker.record_event(sim_time, vehicles[idx1].algo_vehicle, vehicles[idx2].algo_vehicle,
                                     'detection', confidence=0.99, is_crash=False)
    
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

def check_physical_crashes(vehicles: List[Vehicle], crash_tracker: CrashTracker = None,
                          detailed_tracker: DetailedCrashTracker = None, sim_time: float = 0.0):
    """Check if vehicles are actually physically touching (actual crash)"""
    algo_vehicles = [v.algo_vehicle for v in vehicles]
    sim = vehicle_simulation(algo_vehicles)
    
    # Check for physical collision
    is_crash, idx1, idx2 = sim.check_physical_collision(collision_distance=2.0)
    
    # Record actual crash if crash tracker is provided
    if crash_tracker is not None and is_crash and idx1 is not None and idx2 is not None:
        crash_tracker.record_actual_crash(sim_time, idx1, idx2)
    
    # Record detailed crash if detailed tracker is provided
    if detailed_tracker is not None and is_crash and idx1 is not None and idx2 is not None:
        # Add id to vehicles for tracking
        vehicles[idx1].algo_vehicle.id = idx1
        vehicles[idx2].algo_vehicle.id = idx2
        detailed_tracker.record_event(sim_time, vehicles[idx1].algo_vehicle, vehicles[idx2].algo_vehicle,
                                     'crash', confidence=None, is_crash=True)
    
    return is_crash, idx1, idx2

def draw_bplane_panel(screen, vehicles, collision_pair, collision_prob, font, time, is_active=True):
    """Draw b-plane visualization panel (always visible, greyed when inactive)"""
    # Use first two vehicles if no collision pair
    if collision_pair[0] is None or collision_pair[1] is None:
        if len(vehicles) < 2:
            return
        v1 = vehicles[0]
        v2 = vehicles[1]
        collision_prob = 0.0
    else:
        v1 = vehicles[collision_pair[0]]
        v2 = vehicles[collision_pair[1]]
    
    # Panel dimensions and position (left side, below zoom indicator)
    panel_width = 320
    panel_height = 280
    panel_x = 15
    panel_y = WIN_H - panel_height - 75  # Above bottom, below zoom indicator
    
    # Draw panel background with glow effect
    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    
    # Greyed out when inactive
    if not is_active:
        bg_alpha = 200
        border_alpha = 150
        text_brightness = 0.6  # Dim the text
    else:
        bg_alpha = 250
        border_alpha = 220
        text_brightness = 1.0
    
    # Danger glow effect (only when active)
    if is_active and collision_prob > 0.5:
        glow_intensity = int(30 + 20 * np.sin(time * 4))
        pygame.draw.rect(panel_surf, (255, 100, 100, glow_intensity), panel_surf.get_rect(), border_radius=10)
    
    pygame.draw.rect(panel_surf, (30, 35, 45, bg_alpha), panel_surf.get_rect(), border_radius=10)
    
    # Border color based on risk (greyed when inactive)
    if is_active:
        border_color = COLOR_DANGER if collision_prob > 0.5 else COLOR_CAUTION if collision_prob > 0.2 else (80, 90, 110)
    else:
        border_color = (60, 65, 75)
    
    pygame.draw.rect(panel_surf, (*border_color[:3], border_alpha), panel_surf.get_rect(), width=2, border_radius=10)
    screen.blit(panel_surf, (panel_x, panel_y))
    
    # Helper to apply brightness dimming
    def dim_color(color, brightness):
        return tuple(int(c * brightness) for c in color[:3])
    
    # Title with icon
    title_font = pygame.font.SysFont("Arial", 14, bold=True)
    title_color = dim_color((220, 230, 240), text_brightness)
    title_text = "⚠ B-Plane Collision Analysis" if is_active else "⚠ B-Plane Collision Analysis (Standby)"
    title = title_font.render(title_text, True, title_color)
    screen.blit(title, (panel_x + 12, panel_y + 10))
    
    # Divider line
    div_color = dim_color((80, 90, 110), text_brightness)
    pygame.draw.line(screen, div_color, (panel_x + 12, panel_y + 30), (panel_x + panel_width - 12, panel_y + 30), 1)
    
    # Vehicle pair info with colored dots
    pair_y = panel_y + 38
    
    if collision_pair[0] is not None and collision_pair[1] is not None:
        v1_id = collision_pair[0]
        v2_id = collision_pair[1]
    else:
        v1_id = 0
        v2_id = 1
    
    if is_active:
        v1_color = v1.color if not v1.is_collision_risk else COLOR_DANGER
        v2_color = v2.color if not v2.is_collision_risk else COLOR_DANGER
    else:
        v1_color = dim_color(v1.color, 0.5)
        v2_color = dim_color(v2.color, 0.5)
    
    pygame.draw.circle(screen, v1_color, (panel_x + 15, pair_y + 7), 5)
    pygame.draw.circle(screen, v2_color, (panel_x + 95, pair_y + 7), 5)
    
    text_color = dim_color((200, 210, 220), text_brightness)
    pair_text = font.render(f"V{v1_id}", True, text_color)
    screen.blit(pair_text, (panel_x + 25, pair_y + 2))
    
    arrow_color = dim_color((150, 160, 170), text_brightness)
    arrow_text = font.render("↔", True, arrow_color)
    screen.blit(arrow_text, (panel_x + 60, pair_y + 2))
    
    pair_text2 = font.render(f"V{v2_id}", True, text_color)
    screen.blit(pair_text2, (panel_x + 105, pair_y + 2))
    
    # Large Pc display with animated bar
    pc_y = panel_y + 62
    label_color = dim_color((180, 190, 200), text_brightness)
    pc_label = font.render("Collision Probability:", True, label_color)
    screen.blit(pc_label, (panel_x + 15, pc_y))
    
    pc_font = pygame.font.SysFont("Arial", 28, bold=True)
    if is_active:
        pc_color = COLOR_DANGER if collision_prob > 0.5 else COLOR_CAUTION if collision_prob > 0.2 else COLOR_SAFE
    else:
        pc_color = dim_color((100, 110, 120), text_brightness)
    
    pc_text = pc_font.render(f"{collision_prob*100:.1f}%", True, pc_color)
    screen.blit(pc_text, (panel_x + 15, pc_y + 18))
    
    # Probability bar
    bar_x = panel_x + 15
    bar_y = pc_y + 52
    bar_width = panel_width - 30
    bar_height = 12
    
    # Background bar
    bar_bg_color = dim_color((50, 55, 65), text_brightness)
    pygame.draw.rect(screen, bar_bg_color, (bar_x, bar_y, bar_width, bar_height), border_radius=6)
    
    # Filled bar with gradient effect
    fill_width = int(bar_width * collision_prob)
    if fill_width > 0:
        fill_rect = pygame.Rect(bar_x, bar_y, fill_width, bar_height)
        pygame.draw.rect(screen, pc_color, fill_rect, border_radius=6)
    
    # Bar outline
    bar_outline_color = dim_color((100, 110, 120), text_brightness)
    pygame.draw.rect(screen, bar_outline_color, (bar_x, bar_y, bar_width, bar_height), width=1, border_radius=6)
    
    # Draw mini b-plane visualization (enhanced)
    bplane_center_x = panel_x + panel_width // 2
    bplane_center_y = panel_y + 165
    bplane_radius = 50
    
    # B-plane background circle
    pygame.draw.circle(screen, (40, 45, 55), (bplane_center_x, bplane_center_y), bplane_radius + 2)
    
    # Draw coordinate axes with arrowheads
    axis_color = (100, 110, 130)
    # Horizontal axis (ξ)
    pygame.draw.line(screen, axis_color, 
                    (bplane_center_x - bplane_radius - 5, bplane_center_y),
                    (bplane_center_x + bplane_radius + 5, bplane_center_y), 2)
    pygame.draw.polygon(screen, axis_color, [
        (bplane_center_x + bplane_radius + 5, bplane_center_y),
        (bplane_center_x + bplane_radius, bplane_center_y - 3),
        (bplane_center_x + bplane_radius, bplane_center_y + 3)
    ])
    
    # Vertical axis (ζ)
    pygame.draw.line(screen, axis_color,
                    (bplane_center_x, bplane_center_y - bplane_radius - 5),
                    (bplane_center_x, bplane_center_y + bplane_radius + 5), 2)
    pygame.draw.polygon(screen, axis_color, [
        (bplane_center_x, bplane_center_y - bplane_radius - 5),
        (bplane_center_x - 3, bplane_center_y - bplane_radius),
        (bplane_center_x + 3, bplane_center_y - bplane_radius)
    ])
    
    # Draw collision circle (danger zone) with pulsing effect
    collision_radius = int(bplane_radius * 0.28)
    pulse = int(3 * np.sin(time * 3))
    pygame.draw.circle(screen, (*COLOR_DANGER, 100), (bplane_center_x, bplane_center_y), 
                      collision_radius + pulse, 0)
    pygame.draw.circle(screen, COLOR_DANGER, (bplane_center_x, bplane_center_y), 
                      collision_radius, 2)
    
    # UKF Uncertainty visualization
    v1_pos = v1.algo_vehicle.external_state[:2]
    v2_pos = v2.algo_vehicle.external_state[:2]
    rel_pos = np.array(v2_pos) - np.array(v1_pos)
    dist = np.linalg.norm(rel_pos)
    
    if dist > 0:
        # Calculate miss distance in b-plane
        display_scale = min(bplane_radius * 0.6, max(5, 30 / max(dist, 1)))
        display_offset = (rel_pos / dist) * dist * display_scale
        miss_x = int(bplane_center_x + display_offset[0])
        miss_y = int(bplane_center_y - display_offset[1])
        
        # Clamp to visible area
        miss_x = max(bplane_center_x - bplane_radius, min(bplane_center_x + bplane_radius, miss_x))
        miss_y = max(bplane_center_y - bplane_radius, min(bplane_center_y + bplane_radius, miss_y))
        
        # Draw UKF uncertainty ellipse (covariance visualization)
        P1 = v1.algo_vehicle.P[:2, :2]
        P2 = v2.algo_vehicle.P[:2, :2]
        P_combined = P1 + P2
        
        # Draw 1σ, 2σ, 3σ ellipses
        try:
            eigenvalues, eigenvectors = np.linalg.eig(P_combined)
            if np.all(eigenvalues > 0):
                for sigma, alpha_val in [(3, 60), (2, 90), (1, 120)]:
                    ell_w = 2 * sigma * np.sqrt(max(eigenvalues[0], 0)) * display_scale * 2
                    ell_h = 2 * sigma * np.sqrt(max(eigenvalues[1], 0)) * display_scale * 2
                    ell_w = min(ell_w, bplane_radius * 1.5)
                    ell_h = min(ell_h, bplane_radius * 1.5)
                    
                    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                    
                    # Draw ellipse as polygon
                    n_points = 32
                    points = []
                    for i in range(n_points):
                        theta = 2 * np.pi * i / n_points
                        x_local = np.cos(theta) * ell_w / 2
                        y_local = np.sin(theta) * ell_h / 2
                        x_rot = x_local * np.cos(angle) - y_local * np.sin(angle)
                        y_rot = x_local * np.sin(angle) + y_local * np.cos(angle)
                        points.append((miss_x + x_rot, miss_y - y_rot))
                    
                    if len(points) > 2:
                        pygame.draw.polygon(screen, (*COLOR_CAUTION, alpha_val), points, 1)
        except:
            pass
        
        # Draw connecting line from center to miss point
        pygame.draw.line(screen, (120, 140, 160, 150), 
                        (bplane_center_x, bplane_center_y), (miss_x, miss_y), 1)
        
        # Draw miss point with glow
        pygame.draw.circle(screen, (*pc_color, 180), (miss_x, miss_y), 6)
        pygame.draw.circle(screen, pc_color, (miss_x, miss_y), 4)
        pygame.draw.circle(screen, (240, 245, 250), (miss_x, miss_y), 2)
    
    # Axis labels
    axis_font = pygame.font.SysFont("Arial", 10, bold=True)
    x_label = axis_font.render("ξ", True, (160, 170, 180))
    y_label = axis_font.render("ζ", True, (160, 170, 180))
    screen.blit(x_label, (bplane_center_x + bplane_radius + 10, bplane_center_y - 8))
    screen.blit(y_label, (bplane_center_x - 6, bplane_center_y - bplane_radius - 16))
    
    # Statistics section
    stats_y = panel_y + 235
    small_font = pygame.font.SysFont("Arial", 9)
    stats_color = dim_color((160, 170, 180), text_brightness)
    
    # Distance
    dist_m = np.linalg.norm(np.array(v1_pos) - np.array(v2_pos))
    dist_text = small_font.render(f"Distance: {dist_m:.1f}m", True, stats_color)
    screen.blit(dist_text, (panel_x + 15, stats_y))
    
    # Relative velocity
    v1_vel = v1.algo_vehicle.external_state[2]
    v2_vel = v2.algo_vehicle.external_state[2]
    rel_vel_mag = abs(v1_vel - v2_vel) * 3.6  # Convert to km/h
    vel_text = small_font.render(f"Rel. Speed: {rel_vel_mag:.1f}km/h", True, stats_color)
    screen.blit(vel_text, (panel_x + 15, stats_y + 12))
    
    # UKF confidence (trace of covariance)
    P1 = v1.algo_vehicle.P[:2, :2]
    P2 = v2.algo_vehicle.P[:2, :2]
    uncertainty = np.sqrt(np.trace(P1 + P2))
    ukf_text = small_font.render(f"UKF Uncertainty: {uncertainty:.2f}", True, stats_color)
    screen.blit(ukf_text, (panel_x + 180, stats_y))
    
    # Time to closest approach (simplified)
    if abs(v1_vel - v2_vel) > 0.1:
        ttca = dist_m / abs(v1_vel - v2_vel)
        ttca_text = small_font.render(f"TTC: {ttca:.1f}s", True, stats_color)
        screen.blit(ttca_text, (panel_x + 180, stats_y + 12))
    else:
        ttca_text = small_font.render(f"TTC: N/A", True, stats_color)
        screen.blit(ttca_text, (panel_x + 180, stats_y + 12))

def draw_ukf_stats_widget(screen, vehicles, font, sim_time):
    """Draw UKF algorithm statistics widget"""
    widget_width = 200
    widget_height = 160
    widget_x = WIN_W - widget_width - 15
    widget_y = WIN_H - widget_height - 15  # Bottom-right corner
    
    # Background
    widget_surf = pygame.Surface((widget_width, widget_height), pygame.SRCALPHA)
    pygame.draw.rect(widget_surf, (30, 35, 45, 230), widget_surf.get_rect(), border_radius=8)
    pygame.draw.rect(widget_surf, (80, 90, 110, 180), widget_surf.get_rect(), width=2, border_radius=8)
    screen.blit(widget_surf, (widget_x, widget_y))
    
    # Title
    title_font = pygame.font.SysFont("Arial", 12, bold=True)
    title = title_font.render("UKF Algorithm Stats", True, (200, 220, 240))
    screen.blit(title, (widget_x + 10, widget_y + 8))
    
    # Divider
    pygame.draw.line(screen, (80, 90, 110), (widget_x + 10, widget_y + 26), (widget_x + widget_width - 10, widget_y + 26), 1)
    
    # Calculate average uncertainty
    avg_uncertainty = 0.0
    max_uncertainty = 0.0
    for v in vehicles:
        P = v.algo_vehicle.P[:2, :2]
        unc = np.sqrt(np.trace(P))
        avg_uncertainty += unc
        max_uncertainty = max(max_uncertainty, unc)
    
    if len(vehicles) > 0:
        avg_uncertainty /= len(vehicles)
    
    # Display stats
    small_font = pygame.font.SysFont("Arial", 10)
    y_offset = widget_y + 35
    
    # Number of vehicles tracked
    text = small_font.render(f"Vehicles Tracked: {len(vehicles)}", True, (180, 190, 200))
    screen.blit(text, (widget_x + 10, y_offset))
    y_offset += 16
    
    # Average uncertainty
    unc_color = COLOR_SAFE if avg_uncertainty < 0.5 else COLOR_CAUTION if avg_uncertainty < 1.0 else COLOR_DANGER
    text = small_font.render(f"Avg Uncertainty: {avg_uncertainty:.3f}m", True, unc_color)
    screen.blit(text, (widget_x + 10, y_offset))
    
    # Uncertainty bar
    y_offset += 14
    bar_width = widget_width - 20
    bar_height = 6
    pygame.draw.rect(screen, (50, 55, 65), (widget_x + 10, y_offset, bar_width, bar_height), border_radius=3)
    fill_width = int(bar_width * min(avg_uncertainty / 2.0, 1.0))
    if fill_width > 0:
        pygame.draw.rect(screen, unc_color, (widget_x + 10, y_offset, fill_width, bar_height), border_radius=3)
    
    y_offset += 12
    
    # Max uncertainty
    text = small_font.render(f"Max Uncertainty: {max_uncertainty:.3f}m", True, (160, 170, 180))
    screen.blit(text, (widget_x + 10, y_offset))
    y_offset += 16
    
    # Prediction frequency
    text = small_font.render(f"Prediction Rate: {1/PREDICTION_UPDATE_INTERVAL:.1f}Hz", True, (160, 170, 180))
    screen.blit(text, (widget_x + 10, y_offset))
    y_offset += 16
    
    # UKF parameters (sigma points)
    n_sigma = 2 * 4 + 1  # 2*n+1 for n=4 state dimensions
    text = small_font.render(f"Sigma Points: {n_sigma}", True, (160, 170, 180))
    screen.blit(text, (widget_x + 10, y_offset))
    y_offset += 14
    
    # Small visualization of sigma points (conceptual)
    viz_x = widget_x + widget_width // 2
    viz_y = y_offset + 10
    center_radius = 3
    
    # Center point (mean)
    pygame.draw.circle(screen, (120, 180, 220), (viz_x, viz_y), center_radius + 1)
    
    # Sigma points around center (simplified 2D projection)
    sigma_radius = 15
    pulse = int(2 * np.sin(sim_time * 2 * np.pi))
    for i in range(8):
        angle = 2 * np.pi * i / 8
        sx = int(viz_x + (sigma_radius + pulse) * np.cos(angle))
        sy = int(viz_y + (sigma_radius + pulse) * np.sin(angle))
        pygame.draw.circle(screen, (100, 140, 180, 180), (sx, sy), 2)
        pygame.draw.line(screen, (80, 120, 160, 100), (viz_x, viz_y), (sx, sy), 1)

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
    # Load config to get current map
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        map_name = config['current_map']
    except FileNotFoundError:
        print("ERROR: config.json not found. Using default map 'austin'")
        map_name = "austin"
    
    # Load map data using config
    map_json_path = Path(f"data/{map_name}.json")
    if not map_json_path.exists():
        print(f"ERROR: Map file {map_json_path} not found.")
        print(f"Please run: python utils/test.py && python utils/parse_to_json.py")
        sys.exit(1)
    
    print(f"Loading map: {map_name}")
    nodes, edges = load_graph(map_json_path)
    
    # Load building data (optional - will be empty list if not available)
    buildings_path = Path(f"data/{map_name}_buildings.json")
    buildings = load_buildings(buildings_path)
    print(f"Loaded {len(buildings)} buildings")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("B-Plane Collision Prediction — Enhanced Map Navigation")
    clock = pygame.time.Clock()

    w2s, bounds, base_scale, (base_offx, base_offy) = make_projector(nodes, WIN_W, WIN_H, MARGIN)

    scale = 1.0
    pan = [0.0, 0.0]
    dragging = False
    last_mouse = (0, 0)
    show_buildings = len(buildings) > 0  # Show buildings by default if available
    show_minimap = True  # Minimap enabled by default

    G = RoadGraph(nodes, edges)
    vehicles = [Vehicle(G, i) for i in range(NUM_AGENTS)]
    
    # Initialize crash trackers (both for compatibility and detailed tracking)
    crash_tracker = CrashTracker(results_dir="results")
    crash_tracker.start_trial()
    detailed_tracker = DetailedCrashTracker(results_dir="results")
    detailed_tracker.start_trial()
    
    # Collision check timer
    collision_timer = 0.0
    collision_detected = False
    collision_pair = (None, None)
    collision_prob = 0.0
    
    # Animation timing
    sim_time = 0.0
    paused = False
    
    # Trial settings
    max_trial_time = 60.0  # Maximum time per trial (seconds)
    num_trials = 1  # Default: single trial (can be changed for batch mode)
    
    # Check for command line arguments for batch mode
    if len(sys.argv) > 1:
        try:
            num_trials = int(sys.argv[1])
            print(f"[Game] Running {num_trials} trials")
        except ValueError:
            print(f"[Game] Invalid number of trials, using default: 1")
    
    # Keyboard control state
    keys_pressed = set()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if not paused:
            sim_time += dt

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                # Save final summary before exiting
                crash_tracker.end_trial()
                detailed_tracker.end_trial(trial_duration=sim_time)
                crash_tracker.save_final_summary()
                detailed_tracker.save_final_summary()
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 3:
                    dragging = True
                    last_mouse = ev.pos
                elif ev.button == 4:  # Scroll up
                    scale = min(MAX_ZOOM, scale * 1.1)
                elif ev.button == 5:  # Scroll down
                    scale = max(MIN_ZOOM, scale / 1.1)
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
                keys_pressed.add(ev.key)
                
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
                elif ev.key == pygame.K_m:
                    show_minimap = not show_minimap
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
            elif ev.type == pygame.KEYUP:
                keys_pressed.discard(ev.key)
        
        # Continuous keyboard controls for smooth panning and zooming
        if pygame.K_w in keys_pressed or pygame.K_UP in keys_pressed:
            pan[1] += PAN_SPEED
        if pygame.K_s in keys_pressed or pygame.K_DOWN in keys_pressed:
            pan[1] -= PAN_SPEED
        if pygame.K_a in keys_pressed or pygame.K_LEFT in keys_pressed:
            pan[0] += PAN_SPEED
        if pygame.K_d in keys_pressed or pygame.K_RIGHT in keys_pressed:
            pan[0] -= PAN_SPEED
        if pygame.K_q in keys_pressed or pygame.K_MINUS in keys_pressed:
            scale = max(MIN_ZOOM, scale - ZOOM_SPEED)
        if pygame.K_e in keys_pressed or pygame.K_EQUALS in keys_pressed:
            scale = min(MAX_ZOOM, scale + ZOOM_SPEED)

        # Update vehicles (only when not paused)
        if not paused:
            for v in vehicles:
                v.update(SIM_DT)

            # Check for physical crashes (actual collisions)
            physical_crash, crash_idx1, crash_idx2 = check_physical_crashes(vehicles, crash_tracker, detailed_tracker, sim_time)
            
            # Check for collisions periodically (algorithm predictions)
            collision_timer += dt
            if collision_timer >= PREDICTION_UPDATE_INTERVAL:
                collision_timer = 0.0
                collision_detected, idx1, idx2, cprob = check_collisions(vehicles, crash_tracker, detailed_tracker, sim_time)
                if collision_detected:
                    collision_pair = (idx1, idx2)
                    collision_prob = cprob
                else:
                    collision_pair = (None, None)
                    collision_prob = 0.0
            
            # Check if trial should end (time limit reached)
            if sim_time >= max_trial_time:
                crash_tracker.end_trial()
                detailed_tracker.end_trial(trial_duration=sim_time)
                # Reset for next trial or exit
                if crash_tracker.total_trials < num_trials:
                    print(f"[Game] Trial {crash_tracker.total_trials} completed, starting trial {crash_tracker.total_trials + 1}")
                    crash_tracker.start_trial()
                    detailed_tracker.start_trial()
                    vehicles = [Vehicle(G, i) for i in range(NUM_AGENTS)]
                    sim_time = 0.0
                else:
                    crash_tracker.save_final_summary()
                    detailed_tracker.save_final_summary()
                    print(f"\n[Game] Completed {num_trials} trials")
                    running = False
        
        # Calculate time pulse for breathing animation (0 to 1)
        time_pulse = (sim_time % 3.0) / 3.0

        # Draw
        screen.fill(BG)
        
        # Draw buildings first (background layer)
        if show_buildings and buildings:
            draw_buildings(screen, buildings, w2s, scale, pan, scale)
        
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
        panel_font = pygame.font.SysFont("Arial", 12)
        title_font = pygame.font.SysFont("Arial", 15, bold=True)
        
        # Top control panel (expanded for extra lines)
        panel_height = 100
        panel_surf = pygame.Surface((WIN_W, panel_height))
        panel_surf.fill((30, 35, 45))
        panel_surf.set_alpha(240)
        screen.blit(panel_surf, (0, 0))
        
        # Title
        title = title_font.render("Vehicle Collision Prediction Simulation", True, (200, 220, 240))
        screen.blit(title, (15, 10))
        
        # Controls (split into three lines for readability)
        controls1 = panel_font.render("Navigation: WASD/Arrows=Pan | Q/E or -/+=Zoom | Right-drag=Pan | Wheel=Zoom | R=Reset", True, (160, 175, 195))
        controls2 = panel_font.render("View: M=Minimap | B=Buildings | L=Labels | U=Uncertainty | P=Predictions", True, (160, 175, 195))
        controls3 = panel_font.render("Effects: G=Ghosts | T=Trails | V=VelArrows | S=SigmaBands | Space=Pause", True, (160, 175, 195))
        screen.blit(controls1, (15, 30))
        screen.blit(controls2, (15, 46))
        screen.blit(controls3, (15, 62))
        
        # Status indicators
        status_items = []
        status_items.append(f"Zoom: {scale:.2f}x")
        status_items.append(f"Vehicles: {NUM_AGENTS}")
        status_items.append(f"{'⏸ PAUSED' if paused else '▶ Running'}")
        status_items.append(f"FPS: {int(clock.get_fps())}")
        
        status_text = " | ".join(status_items)
        status_render = panel_font.render(status_text, True, (140, 160, 180))
        screen.blit(status_render, (15, 80))
        
        # B-plane visualization panel (always visible, greyed when no collision)
        is_collision_active = (collision_detected and collision_pair[0] is not None and collision_pair[1] is not None)
        draw_bplane_panel(screen, vehicles, collision_pair, collision_prob, panel_font, sim_time, is_collision_active)
        
        # Minimap (top-right, below control panel)
        if show_minimap:
            draw_minimap(screen, nodes, edges, vehicles, bounds, pan, scale)
        
        # UKF Stats Widget (right side, below minimap)
        draw_ukf_stats_widget(screen, vehicles, panel_font, sim_time)
        
        # Zoom indicator (bottom-left corner)
        zoom_indicator_x = 15
        zoom_indicator_y = WIN_H - 60
        zoom_surf = pygame.Surface((120, 50), pygame.SRCALPHA)
        pygame.draw.rect(zoom_surf, (30, 35, 45, 220), zoom_surf.get_rect(), border_radius=6)
        pygame.draw.rect(zoom_surf, (80, 90, 110, 180), zoom_surf.get_rect(), width=2, border_radius=6)
        screen.blit(zoom_surf, (zoom_indicator_x, zoom_indicator_y))
        
        # Zoom text
        zoom_title = panel_font.render("Zoom Level", True, (180, 190, 200))
        screen.blit(zoom_title, (zoom_indicator_x + 10, zoom_indicator_y + 8))
        
        # Zoom value with color coding
        zoom_color = (120, 180, 220) if scale >= 1.0 else (255, 200, 120)
        zoom_value = title_font.render(f"{scale:.2f}x", True, zoom_color)
        screen.blit(zoom_value, (zoom_indicator_x + 10, zoom_indicator_y + 25))
        
        # Zoom level description
        zoom_desc_font = pygame.font.SysFont("Arial", 9)
        if scale < 0.7:
            zoom_desc = "Major Roads"
        elif scale < 1.2:
            zoom_desc = "Main Roads"
        elif scale < 1.8:
            zoom_desc = "Local Roads"
        else:
            zoom_desc = "All Roads"
        zoom_desc_text = zoom_desc_font.render(zoom_desc, True, (140, 150, 160))
        screen.blit(zoom_desc_text, (zoom_indicator_x + 75, zoom_indicator_y + 31))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
