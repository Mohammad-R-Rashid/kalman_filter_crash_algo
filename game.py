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
AGENT_COLORS = [(80, 200, 255), (255, 140, 90), (180, 255, 140), (250, 220, 80)]

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
ELLIPSE_SIGMA = 2.0  # draw 2-sigma ellipses

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
def draw_uncertainty_ellipse(surf, center_x: int, center_y: int, 
                             P_2x2: np.ndarray, sigma: float, color, scale: float):
    """Draw uncertainty ellipse from 2x2 position covariance matrix"""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P_2x2)
        if np.any(eigenvalues < 0):
            return
        
        width = 2 * sigma * math.sqrt(max(eigenvalues[0], 0)) * scale
        height = 2 * sigma * math.sqrt(max(eigenvalues[1], 0)) * scale
        angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        points = []
        n_points = 32
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            x_local = math.cos(theta) * width / 2
            y_local = math.sin(theta) * height / 2
            x_rot = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_rot = x_local * math.sin(angle) + y_local * math.cos(angle)
            points.append((center_x + x_rot, center_y - y_rot))
        
        if len(points) > 2:
            pygame.draw.polygon(surf, color, points, 1)
    except:
        pass

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
        # Compute control inputs
        accel, steering = self._compute_control_inputs()
        
        # Update the algo vehicle (this handles UKF internally)
        self.algo_vehicle.update_external([accel, steering], dt)
        
        # Update road position
        x, y, v, psi = self.algo_vehicle.external_state
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
    
    def draw(self, surf, world_to_screen, scale, pan, base_scale, font):
        """Draw vehicle with uncertainty ellipse and label"""
        x, y, v, psi = self.algo_vehicle.external_state
        lon, lat = self.m_to_lonlat(x, y)
        px, py = world_to_screen(lon, lat, scale=scale, pan=pan)
        
        # Color based on collision risk
        color = self.color
        is_high_risk = self.is_collision_risk
        if is_high_risk:
            color = (255, 80, 80)  # Red for collision risk
        
        # Draw vehicle as arrow
        tip = (px, py)
        tail = (px - ARROW_LEN * math.cos(psi), py + ARROW_LEN * math.sin(psi))
        left = (tip[0] - ARROW_LEN * math.cos(psi - ARROW_ANG),
                tip[1] + ARROW_LEN * math.sin(psi - ARROW_ANG))
        right = (tip[0] - ARROW_LEN * math.cos(psi + ARROW_ANG),
                 tip[1] + ARROW_LEN * math.sin(psi + ARROW_ANG))
        
        # Outline
        outline_width = 6 if is_high_risk else 5
        pygame.draw.line(surf, (40, 40, 50), tail, tip, outline_width)
        pygame.draw.line(surf, (40, 40, 50), tip, left, outline_width)
        pygame.draw.line(surf, (40, 40, 50), tip, right, outline_width)
        
        # Body
        line_width = 4 if is_high_risk else 3
        pygame.draw.line(surf, color, tail, tip, line_width)
        pygame.draw.line(surf, color, tip, left, line_width)
        pygame.draw.line(surf, color, tip, right, line_width)
        
        # Label
        if SHOW_LABELS:
            speed_kmh = v * 3.6
            label_text = font.render(f"V{self.id} ({speed_kmh:.0f}km/h)", True, (40, 40, 50))
            label_rect = label_text.get_rect(center=(px, py - 22))
            
            bg_rect = label_rect.inflate(6, 3)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.fill((255, 255, 255))
            bg_surface.set_alpha(220)
            surf.blit(bg_surface, bg_rect)
            surf.blit(label_text, label_rect)
            
            # Lane indicator
            if USE_LANES and self.num_lanes > 1:
                lane_color = (100, 150, 200)
                lane_indicator_y = py - 35
                for i in range(self.num_lanes):
                    dot_x = px - (self.num_lanes - 1) * 3 + i * 6
                    if i == self.lane:
                        pygame.draw.circle(surf, lane_color, (dot_x, lane_indicator_y), 3)
                    else:
                        pygame.draw.circle(surf, (180, 180, 180), (dot_x, lane_indicator_y), 2)
        
        # Uncertainty ellipse
        if SHOW_UNCERTAINTY:
            P_2x2 = self.algo_vehicle.P[:2, :2]
            draw_uncertainty_ellipse(surf, px, py, P_2x2, ELLIPSE_SIGMA, color, base_scale * scale)
        
        # Draw predictions
        if SHOW_PREDICTIONS:
            pred_x = self.algo_vehicle.pred_next_state[0]
            pred_y = self.algo_vehicle.pred_next_state[1]
            if pred_x != 0 or pred_y != 0:
                pred_lon, pred_lat = self.m_to_lonlat(pred_x, pred_y)
                pred_px, pred_py = world_to_screen(pred_lon, pred_lat, scale=scale, pan=pan)
                pygame.draw.circle(surf, color, (pred_px, pred_py), 4)
                pygame.draw.line(surf, color + (100,) if len(color) == 3 else color, (px, py), (pred_px, pred_py), 1)
                
                # Draw prediction uncertainty
                P_pred_2x2 = self.algo_vehicle.predicted_P[:2, :2]
                draw_uncertainty_ellipse(surf, pred_px, pred_py, P_pred_2x2, ELLIPSE_SIGMA, 
                                        tuple(int(c * 0.5) for c in color), base_scale * scale)

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
    
    # Make sure all vehicles have predictions (ukf_predict needs to be called)
    for v in vehicles:
        v.algo_vehicle.ukf_predict(SIM_DT)
    
    # Use vehicle_simulation to check for collisions
    algo_vehicles = [v.algo_vehicle for v in vehicles]
    sim = vehicle_simulation(algo_vehicles)
    
    # Check collision at 95% confidence level
    is_collision, idx1, idx2 = sim.is_collision(0.95)
    
    if is_collision and idx1 is not None and idx2 is not None:
        vehicles[idx1].is_collision_risk = True
        vehicles[idx2].is_collision_risk = True
        return True, idx1, idx2
    
    return False, None, None

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
    
    # Collision check timer
    collision_timer = 0.0
    collision_detected = False
    collision_pair = (None, None)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

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
            v.update(SIM_DT)

        # Check for collisions periodically
        collision_timer += dt
        if collision_timer >= PREDICTION_UPDATE_INTERVAL:
            collision_timer = 0.0
            collision_detected, idx1, idx2 = check_collisions(vehicles)
            if collision_detected:
                collision_pair = (idx1, idx2)

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
        if collision_detected and collision_pair[0] is not None:
            warning_panel_width = 260
            warning_panel_height = 80
            warning_surf = pygame.Surface((warning_panel_width, warning_panel_height))
            warning_surf.fill((45, 30, 30))
            warning_surf.set_alpha(240)
            screen.blit(warning_surf, (WIN_W - warning_panel_width - 15, 85))
            
            warning_title = title_font.render("⚠ Collision Warning", True, (255, 220, 220))
            screen.blit(warning_title, (WIN_W - warning_panel_width - 10, 90))
            
            warning_font = pygame.font.SysFont("Arial", 12, bold=True)
            msg = f"V{collision_pair[0]} ↔ V{collision_pair[1]}"
            warning_render = warning_font.render(msg, True, (255, 100, 100))
            screen.blit(warning_render, (WIN_W - warning_panel_width - 5, 115))
            
            detail = panel_font.render("B-plane collision detected!", True, (255, 180, 100))
            screen.blit(detail, (WIN_W - warning_panel_width - 5, 135))
        
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
