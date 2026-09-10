# ================================================================================
# SINGLE-FILE MERGED SIH POTHOLE DETECTION PROJECT
# ================================================================================
# This file combines the project modules into one source file.
# Module boundaries are retained as comments for readability.
#
# The original project used a package layout (config.py + src/*.py + scripts).
# This merged version places definitions in one global namespace and removes
# internal `src.*`/`config` imports so it can be kept as a single source file.
# ================================================================================


# ==============================================================================
# EMBEDDED MODULE: config.py
# ==============================================================================

"""
================================================================================
 config.py  --  Single source of truth for every tunable knob in the system.
================================================================================
Nothing in this project hard-codes a magic number. Everything lives here so the
pipeline can be re-tuned for a new city, a new camera, or a new vehicle without
touching detection / clustering / rendering code.

Author : Pothole-AI prototype
Python : 3.9+
================================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
MODEL_DIR: Path = ROOT_DIR / "models"

for _d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Input / output artefacts -----------------------------------------------------
INPUT_VIDEO: Path = DATA_DIR / "input_dashcam.mp4"
OUTPUT_VIDEO: Path = OUTPUT_DIR / "output_annotated.mp4"
LOG_CSV: Path = OUTPUT_DIR / "pothole_log.csv"
CLUSTER_CSV: Path = OUTPUT_DIR / "pothole_clusters.csv"
MAP_HTML: Path = OUTPUT_DIR / "pothole_map.html"
ALERT_LOG: Path = OUTPUT_DIR / "alerts.log"

# ------------------------------------------------------------------------------
# 2. DETECTION SETTINGS
# ------------------------------------------------------------------------------
# Path to a trained YOLOv8 weights file. If this file does not exist (or the
# `ultralytics` package is not installed) the pipeline transparently falls back
# to the deterministic SimulatedDetector so the prototype ALWAYS runs.
YOLO_WEIGHTS: Path = MODEL_DIR / "pothole_yolov8.pt"

# Pretrained checkpoint used as the starting point for transfer learning.
YOLO_BASE_CHECKPOINT: str = "yolov8s.pt"

CONF_THRESHOLD: float = 0.35      # minimum confidence to accept a detection
IOU_THRESHOLD: float = 0.45       # NMS IoU threshold
MAX_DETECTIONS: int = 30          # per-frame cap (a dashcam frame rarely has more)
POTHOLE_CLASS_NAMES: tuple = ("pothole", "Pothole", "POTHOLE", "damage", "D40")

# Force the simulated detector even if real weights exist (handy for demos).
FORCE_SIMULATION: bool = os.getenv("POTHOLE_FORCE_SIM", "0") == "1"

# ------------------------------------------------------------------------------
# 3. CAMERA / GEOMETRY MODEL
# ------------------------------------------------------------------------------
# A monocular dashcam cannot measure true depth. We therefore build a *relative*
# depth proxy from a pinhole camera + flat-ground assumption. These constants
# describe the physical mounting of the camera and are the only inputs the
# geometry needs.
CAMERA_HEIGHT_M: float = 1.25     # lens height above the road surface (metres)
CAMERA_HFOV_DEG: float = 78.0     # horizontal field of view of a typical dashcam
HORIZON_RATIO: float = 0.45       # horizon sits at 45% of frame height from top

# Depth proxy calibration. `DEPTH_GAIN_CM` is the depth (cm) assigned to a
# pothole whose *physical* footprint is DEPTH_REF_AREA_M2 square metres.
# Empirically, Indian urban potholes follow roughly d ~ 9 * sqrt(area).
DEPTH_GAIN_CM: float = 9.0
DEPTH_REF_AREA_M2: float = 1.0
DEPTH_MIN_CM: float = 0.8
DEPTH_MAX_CM: float = 28.0

# Detections whose bottom edge is above this ratio are too far away for the
# ground-plane maths to be trustworthy, so we discard them.
MIN_BOTTOM_RATIO: float = 0.50

# ------------------------------------------------------------------------------
# 4. VEHICLE PROFILES  --  drives the Impact Severity Index
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class VehicleProfile:
    """Physical properties of a vehicle that decide how badly a pothole hurts."""
    name: str
    ground_clearance_cm: float   # chassis-to-road distance
    tyre_diameter_cm: float      # bigger wheels 'bridge' potholes more easily
    suspension_travel_cm: float  # how much the damper can absorb
    mass_kg: float               # heavier vehicle -> higher impact energy

    @property
    def bridging_capacity_cm(self) -> float:
        """
        Approximate pothole depth this vehicle can absorb without a hard hit.
        A larger wheel drops less far into a hole of a given width, and the
        suspension soaks up part of what remains.
        """
        return 0.06 * self.tyre_diameter_cm + 0.35 * self.suspension_travel_cm


VEHICLE_PROFILES: dict = {
    "Hatchback": VehicleProfile("Hatchback", 16.5, 58.0, 12.0, 1050),
    "Sedan":     VehicleProfile("Sedan",     15.0, 63.0, 13.0, 1350),
    "SUV":       VehicleProfile("SUV",       21.0, 71.0, 18.0, 1800),
    "Truck":     VehicleProfile("Truck",     28.0, 95.0, 22.0, 7500),
    "Two-Wheeler": VehicleProfile("Two-Wheeler", 16.0, 45.0, 10.0, 160),
}

# >>> CHANGE THIS ONE LINE TO SEE SEVERITY SHIFT ACROSS THE WHOLE RUN <<<
VEHICLE_TYPE: str = "Hatchback"

# ------------------------------------------------------------------------------
# 5. IMPACT SEVERITY INDEX (ISI) WEIGHTS
# ------------------------------------------------------------------------------
# ISI = w_d * depth_term + w_a * area_term + w_s * speed_term   (range 0..~1.3)
ISI_WEIGHT_DEPTH: float = 0.55
ISI_WEIGHT_AREA: float = 0.25
ISI_WEIGHT_SPEED: float = 0.20

ISI_AREA_SATURATION_M2: float = 1.2   # area at which the area term maxes out
ISI_SPEED_SATURATION_KMPH: float = 60.0

SEVERITY_THRESHOLDS: dict = {"Low": 0.35, "Medium": 0.62}  # else -> "High"

SEVERITY_COLORS_BGR: dict = {           # OpenCV uses BGR ordering
    "Low":    (80, 200, 60),
    "Medium": (30, 165, 245),
    "High":   (40, 40, 220),
}
SEVERITY_COLORS_HEX: dict = {           # folium uses hex / named colours
    "Low":    "green",
    "Medium": "orange",
    "High":   "red",
}

# ------------------------------------------------------------------------------
# 6. GPS SIMULATION
# ------------------------------------------------------------------------------
# A short drive along Bengaluru's Outer Ring Road. Replace with a real NMEA /
# GPX feed in production -- see src/gps_simulator.py::GPSTrack.from_gpx().
ROUTE_WAYPOINTS: list = [
    (12.935200, 77.614900),
    (12.936800, 77.618300),
    (12.938900, 77.621700),
    (12.941500, 77.624400),
    (12.944300, 77.626100),
    (12.947200, 77.627000),
]
VEHICLE_SPEED_KMPH: float = 34.0   # simulated cruising speed
GPS_NOISE_METRES: float = 1.4      # consumer-grade GPS jitter (1-sigma)
GPS_RANDOM_SEED: int = 42          # reproducible runs

# ------------------------------------------------------------------------------
# 7. SPATIAL CLUSTERING (DBSCAN)
# ------------------------------------------------------------------------------
CLUSTER_RADIUS_M: float = 3.0   # merge detections within 3 metres
CLUSTER_MIN_SAMPLES: int = 1    # a single sighting still becomes a marker
EARTH_RADIUS_M: float = 6_371_008.8

# A cluster seen this many times or more is flagged "verified" on the map.
VERIFIED_MIN_HITS: int = 3

# ------------------------------------------------------------------------------
# 8. ALERTS & RENDERING
# ------------------------------------------------------------------------------
ALERT_SEVERITIES: tuple = ("High",)     # which severities trigger a warning
ALERT_COOLDOWN_S: float = 2.5           # don't spam the driver
ENABLE_AUDIO_ALERT: bool = True         # degrades to silent if no audio backend
ALERT_BEEP_HZ: int = 880
ALERT_BEEP_MS: int = 260

DRAW_HUD: bool = True                   # speed / GPS / counter overlay
DRAW_HORIZON_LINE: bool = False         # debugging aid for the geometry model
OUTPUT_FOURCC: str = "mp4v"

# ------------------------------------------------------------------------------
# 9. SAMPLE VIDEO GENERATOR (used when no real dashcam clip is supplied)
# ------------------------------------------------------------------------------
SAMPLE_VIDEO_WIDTH: int = 960
SAMPLE_VIDEO_HEIGHT: int = 540
SAMPLE_VIDEO_FPS: int = 24
SAMPLE_VIDEO_SECONDS: int = 20
SAMPLE_POTHOLE_COUNT: int = 24          # distinct potholes seeded on the road


def get_vehicle() -> VehicleProfile:
    """Resolve VEHICLE_TYPE into a VehicleProfile, with a helpful error."""
    try:
        return VEHICLE_PROFILES[VEHICLE_TYPE]
    except KeyError as exc:
        valid = ", ".join(VEHICLE_PROFILES)
        raise ValueError(
            f"Unknown VEHICLE_TYPE {VEHICLE_TYPE!r}. Valid options: {valid}"
        ) from exc

# ==============================================================================
# EMBEDDED MODULE: depth_impact.py
# ==============================================================================

# BEGIN FILE: depth_impact.py
########################################################################

"""
================================================================================
 depth_impact.py  --  Monocular geometry, depth proxy, and Impact Severity Index
================================================================================

WHY THIS MODULE EXISTS
----------------------
A YOLO box tells you *where* a pothole is in pixels. It tells you nothing about
how much it will hurt. This module converts pixels -> metres -> a single
actionable severity label, using three stages:

    Stage 1  PROJECTION   pixel box  ->  real-world footprint (m^2) + distance
    Stage 2  DEPTH PROXY  footprint  ->  estimated depth (cm)
    Stage 3  IMPACT       depth + area + speed + vehicle  ->  ISI -> Low/Med/High

HONEST LIMITATION (state this in any viva / demo)
-------------------------------------------------
A single camera CANNOT observe true depth -- the projection of a 2 cm dish and a
20 cm crater can be pixel-identical. Stage 2 is therefore an *empirical proxy*,
not a measurement. It is calibrated on the well-documented observation that
pothole depth scales roughly with the square root of surface area. For true
depth you need stereo, LiDAR, an IMU-based jolt sensor, or a learned monocular
depth network (MiDaS / Depth-Anything). The code is structured so Stage 2 can be
swapped out without touching Stages 1 and 3.
================================================================================
"""

import math
from dataclasses import dataclass, asdict
from typing import Tuple
from config import VehicleProfile


# ==============================================================================
# STAGE 1 -- CAMERA PROJECTION
# ==============================================================================
class GroundPlaneProjector:
    """
    Pinhole camera + flat-road model.

    Assumptions (documented deliberately -- every one of them is a failure mode):
      * The road ahead is a flat plane.
      * The camera's optical axis is parallel to that plane (no pitch).
      * Lens distortion has already been corrected, or is negligible.

    Under those assumptions, a point on the road that appears `dy` pixels *below*
    the horizon line lies at a ground distance of:

            Z = f * H_cam / dy

    where `f` is the focal length in pixels and `H_cam` the camera height. Once
    Z is known, any pixel length `L_px` at that distance maps to a real length:

            L_m = L_px * Z / f
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h

        # Focal length in pixels, derived from the horizontal field of view.
        #   tan(HFOV/2) = (w/2) / f
        half_fov = math.radians(config.CAMERA_HFOV_DEG) / 2.0
        self.focal_px = (frame_w / 2.0) / math.tan(half_fov)

        # y-coordinate of the horizon in pixels.
        self.horizon_y = frame_h * config.HORIZON_RATIO

    # --------------------------------------------------------------------
    def distance_to_row(self, y_px: float) -> float:
        """Ground distance (metres) to the road point imaged at row `y_px`."""
        dy = y_px - self.horizon_y
        if dy <= 1e-6:                      # at or above the horizon -> infinity
            return float("inf")
        return (self.focal_px * config.CAMERA_HEIGHT_M) / dy

    # --------------------------------------------------------------------
    def project_box(self, x1: float, y1: float, x2: float, y2: float
                    ) -> Tuple[float, float, float, float]:
        """
        Convert a pixel bounding box into real-world dimensions and position.

        Returns
        -------
        (width_m, length_m, distance_m, lateral_m)
            width_m    : cross-road extent of the pothole
            length_m   : along-road extent (foreshortened, so corrected below)
            distance_m : range from the camera to the near edge
            lateral_m  : sideways offset from the optical axis, +ve to the right.
                         This is what lets us geolocate the POTHOLE rather than
                         the car -- see gps_simulator.project_pothole().
        """
        # Distance is evaluated at the *bottom* edge: that is where the pothole
        # meets the road plane, so it is the only row we can trust.
        dist_near = self.distance_to_row(y2)
        dist_far = self.distance_to_row(y1)

        if not math.isfinite(dist_near):
            return 0.0, 0.0, float("inf"), 0.0

        # Cross-road width: a simple pixel-to-metre scale at that distance.
        width_m = (x2 - x1) * dist_near / self.focal_px

        # Along-road length: the difference of the two ground distances. This
        # automatically handles perspective foreshortening -- a pothole far away
        # occupies few pixels vertically but many metres of road.
        if math.isfinite(dist_far):
            length_m = max(dist_far - dist_near, 0.0)
        else:
            length_m = width_m          # degenerate case: assume roughly square

        # Sideways offset of the box centre from the optical axis.
        cx = (x1 + x2) / 2.0
        lateral_m = (cx - self.w / 2.0) * dist_near / self.focal_px

        # Clamp against physically absurd values caused by detector noise.
        width_m = min(width_m, 6.0)
        length_m = min(length_m, 6.0)
        return width_m, length_m, dist_near, lateral_m


# ==============================================================================
# STAGE 2 + 3 -- THE RESULT OBJECT
# ==============================================================================
@dataclass
class ImpactAssessment:
    """Everything the system knows about one detected pothole, in real units."""
    width_m: float
    length_m: float
    area_m2: float
    distance_m: float
    lateral_m: float           # +ve = right of the vehicle's centreline
    depth_cm: float
    isi: float                 # Impact Severity Index, 0 .. ~1.3
    severity: str              # "Low" | "Medium" | "High"
    vehicle: str

    def as_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# STAGE 2 -- DEPTH PROXY
# ==============================================================================
def estimate_depth_cm(area_m2: float, aspect_ratio: float) -> float:
    """
    Empirical depth proxy.

    Two signals feed it:

    1. AREA. Road-surface studies consistently show depth growing with the
       square root of a pothole's plan area -- water pooling and repeated wheel
       loading erode a hole outward and downward together. So:

            depth ~ DEPTH_GAIN_CM * sqrt(area / reference_area)

    2. ASPECT RATIO. A compact, near-circular hole is typically a deep punch-out;
       a long thin strip is usually shallow surface ravelling or a crack seam.
       We therefore apply a mild penalty to elongated shapes.

    Parameters
    ----------
    area_m2 : float
        Real-world plan area of the pothole from Stage 1.
    aspect_ratio : float
        max(w, l) / min(w, l), always >= 1.

    Returns
    -------
    float
        Estimated depth in centimetres, clamped to a physically sane range.
    """
    if area_m2 <= 0:
        return config.DEPTH_MIN_CM

    # --- primary term: sqrt scaling with area -----------------------------
    base = config.DEPTH_GAIN_CM * math.sqrt(area_m2 / config.DEPTH_REF_AREA_M2)

    # --- shape correction --------------------------------------------------
    # aspect 1.0 -> factor 1.00 (compact, deep)
    # aspect 4.0 -> factor ~0.70 (elongated, shallow)
    compactness = 1.0 / max(aspect_ratio, 1.0)
    shape_factor = 0.6 + 0.4 * compactness

    depth = base * shape_factor
    return float(min(max(depth, config.DEPTH_MIN_CM), config.DEPTH_MAX_CM))


# ==============================================================================
# STAGE 3 -- IMPACT SEVERITY INDEX
# ==============================================================================
def compute_isi(depth_cm: float,
                area_m2: float,
                speed_kmph: float,
                vehicle: VehicleProfile) -> float:
    """
    Impact Severity Index -- a weighted blend of three normalised risk terms.

        ISI = w_d * depth_term + w_a * area_term + w_s * speed_term

    DEPTH TERM (the dominant one, weight 0.55)
        Depth is scored *relative to what this vehicle can absorb*, not in
        absolute centimetres. A 12 cm pothole is a suspension-breaker for a
        hatchback and a minor thud for a truck. The reference is the sum of the
        vehicle's ground clearance and its bridging capacity (wheel size +
        suspension travel):

            depth_term = depth / (clearance + bridging_capacity)

        This is why swapping VEHICLE_TYPE re-scores the entire dataset.

    AREA TERM (weight 0.25)
        A wide pothole cannot be straddled or swerved around, and guarantees
        both wheels on one axle drop in. Saturates at ISI_AREA_SATURATION_M2.

    SPEED TERM (weight 0.20)
        Impact energy rises with the square of velocity, so the normalised speed
        is squared before weighting.

    Returns
    -------
    float
        Unclipped index. Roughly 0 (harmless) to 1.3 (axle damage likely).
    """
    # ---- depth term -------------------------------------------------------
    absorbable_cm = vehicle.ground_clearance_cm + vehicle.bridging_capacity_cm
    depth_term = depth_cm / max(absorbable_cm, 1e-6)

    # ---- area term --------------------------------------------------------
    area_term = min(area_m2 / config.ISI_AREA_SATURATION_M2, 1.0)

    # ---- speed term (energy ~ v^2) ---------------------------------------
    v_norm = min(speed_kmph / config.ISI_SPEED_SATURATION_KMPH, 1.0)
    speed_term = v_norm ** 2

    isi = (config.ISI_WEIGHT_DEPTH * depth_term
           + config.ISI_WEIGHT_AREA * area_term
           + config.ISI_WEIGHT_SPEED * speed_term)
    return float(isi)


def classify_severity(isi: float) -> str:
    """Map a continuous ISI onto the three operational buckets."""
    if isi < config.SEVERITY_THRESHOLDS["Low"]:
        return "Low"
    if isi < config.SEVERITY_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


# ==============================================================================
# PUBLIC ENTRY POINT -- one call does all three stages
# ==============================================================================
def assess(box_xyxy: Tuple[float, float, float, float],
           projector: GroundPlaneProjector,
           speed_kmph: float,
           vehicle: VehicleProfile) -> ImpactAssessment | None:
    """
    Run the full pixels -> severity pipeline for a single bounding box.

    Parameters
    ----------
    box_xyxy : (x1, y1, x2, y2) in pixel coordinates.
    projector : a GroundPlaneProjector matched to the frame size.
    speed_kmph : the vehicle's speed at this timestamp.
    vehicle : the VehicleProfile being simulated.

    Returns
    -------
    ImpactAssessment, or None if the box is too far away / above the horizon to
    be measured reliably.
    """
    x1, y1, x2, y2 = box_xyxy

    # Reject detections whose bottom edge sits too high in the frame: the
    # ground-plane maths becomes numerically unstable near the horizon.
    if y2 < projector.h * config.MIN_BOTTOM_RATIO:
        return None

    width_m, length_m, distance_m, lateral_m = projector.project_box(x1, y1, x2, y2)
    if width_m <= 0 or length_m <= 0 or not math.isfinite(distance_m):
        return None

    area_m2 = width_m * length_m
    aspect = max(width_m, length_m) / max(min(width_m, length_m), 1e-6)

    depth_cm = estimate_depth_cm(area_m2, aspect)
    isi = compute_isi(depth_cm, area_m2, speed_kmph, vehicle)

    return ImpactAssessment(
        width_m=round(width_m, 3),
        length_m=round(length_m, 3),
        area_m2=round(area_m2, 4),
        distance_m=round(distance_m, 2),
        lateral_m=round(lateral_m, 2),
        depth_cm=round(depth_cm, 2),
        isi=round(isi, 4),
        severity=classify_severity(isi),
        vehicle=vehicle.name,
    )


########################################################################
# END FILE: depth_impact.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: gps_simulator.py
# ==============================================================================

# BEGIN FILE: gps_simulator.py
########################################################################

"""
================================================================================
 gps_simulator.py  --  Mock GNSS feed synchronised to video timestamps
================================================================================

In a real deployment the dashcam emits an NMEA/GPX stream and every frame is
tagged with a fix. For the prototype we synthesise that stream: the vehicle
drives a fixed polyline at a fixed speed, and `GPSTrack.fix_at(t)` returns the
position at any video timestamp `t`.

The output is deliberately imperfect. Consumer GNSS has a 1-3 metre error, and
that error is exactly why the DBSCAN clustering stage exists -- the same
physical pothole seen in 8 consecutive frames produces 8 slightly different
coordinates that must be merged back into one map marker.

Swapping in real data
---------------------
    track = GPSTrack.from_gpx("drive.gpx")     # instead of GPSTrack.from_config()
Everything downstream is unchanged.
================================================================================
"""

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple
# ==============================================================================
# GEODESY HELPERS
# ==============================================================================
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) pairs."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * config.EARTH_RADIUS_M * math.asin(math.sqrt(h))


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Initial compass bearing (degrees, 0 = North) travelling from a to b."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def offset_metres(point: Tuple[float, float],
                  north_m: float,
                  east_m: float) -> Tuple[float, float]:
    """
    Shift a (lat, lon) by a north/east offset in metres.

    Uses the equirectangular approximation, which is accurate to well under a
    centimetre for the sub-100-metre offsets we deal with.
    """
    lat, lon = point
    dlat = north_m / config.EARTH_RADIUS_M
    dlon = east_m / (config.EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


# ==============================================================================
# THE FIX OBJECT
# ==============================================================================
@dataclass
class GPSFix:
    """One GNSS sample."""
    timestamp_s: float
    latitude: float
    longitude: float
    speed_kmph: float
    heading_deg: float


# ==============================================================================
# THE TRACK
# ==============================================================================
class GPSTrack:
    """
    A drivable route with constant-speed interpolation and realistic jitter.

    Internally the waypoint polyline is converted into a cumulative-distance
    table. Looking up a timestamp becomes: distance = speed * t, then a binary
    search into that table plus linear interpolation on the matching segment.
    """

    def __init__(self,
                 waypoints: Sequence[Tuple[float, float]],
                 speed_kmph: float = config.VEHICLE_SPEED_KMPH,
                 noise_m: float = config.GPS_NOISE_METRES,
                 seed: int = config.GPS_RANDOM_SEED):
        if len(waypoints) < 2:
            raise ValueError("A route needs at least two waypoints.")

        self.waypoints: List[Tuple[float, float]] = list(waypoints)
        self.speed_kmph = speed_kmph
        self.speed_mps = speed_kmph / 3.6
        self.noise_m = noise_m
        self._rng = random.Random(seed)

        # Cumulative distance along the polyline: cum[i] = metres from start to
        # waypoint i. cum[-1] is the total route length.
        self._cum: List[float] = [0.0]
        for i in range(1, len(self.waypoints)):
            step = haversine_m(self.waypoints[i - 1], self.waypoints[i])
            self._cum.append(self._cum[-1] + step)

        self.total_length_m = self._cum[-1]
        self.duration_s = self.total_length_m / max(self.speed_mps, 1e-6)

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_config(cls) -> "GPSTrack":
        """Build the default demo route defined in config.ROUTE_WAYPOINTS."""
        return cls(config.ROUTE_WAYPOINTS)

    @classmethod
    def from_gpx(cls, gpx_path: str, **kwargs) -> "GPSTrack":
        """
        Load a real recorded drive from a .gpx file.

        Kept dependency-free by parsing the XML directly -- a GPX track is just
        a list of <trkpt lat="..." lon="..."> elements.
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(gpx_path)
        pts: List[Tuple[float, float]] = []
        for el in tree.iter():
            if el.tag.endswith("trkpt") or el.tag.endswith("rtept"):
                pts.append((float(el.attrib["lat"]), float(el.attrib["lon"])))
        if len(pts) < 2:
            raise ValueError(f"No usable track points found in {gpx_path}")
        return cls(pts, **kwargs)

    # ------------------------------------------------------------------ lookup
    def _point_at_distance(self, dist_m: float) -> Tuple[Tuple[float, float], float]:
        """
        Interpolate a position at `dist_m` along the route.

        Returns ((lat, lon), heading_deg). If the route is shorter than the
        video, we loop back to the start so the demo never runs out of road.
        """
        if self.total_length_m <= 0:
            return self.waypoints[0], 0.0

        dist_m = dist_m % self.total_length_m   # wrap around

        # Find the segment containing dist_m.
        seg = 0
        for i in range(1, len(self._cum)):
            if dist_m <= self._cum[i]:
                seg = i - 1
                break
        else:
            seg = len(self.waypoints) - 2

        a, b = self.waypoints[seg], self.waypoints[seg + 1]
        seg_len = self._cum[seg + 1] - self._cum[seg]
        frac = 0.0 if seg_len <= 0 else (dist_m - self._cum[seg]) / seg_len

        lat = a[0] + (b[0] - a[0]) * frac
        lon = a[1] + (b[1] - a[1]) * frac
        return (lat, lon), bearing_deg(a, b)

    # --------------------------------------------------------------------------
    def fix_at(self, timestamp_s: float, jitter: bool = True) -> GPSFix:
        """
        Return the simulated GNSS fix for a given video timestamp.

        `jitter=True` adds zero-mean Gaussian noise of GPS_NOISE_METRES sigma in
        both the north and east directions -- this is what makes the clustering
        stage a real problem rather than a no-op.
        """
        travelled = self.speed_mps * timestamp_s
        (lat, lon), heading = self._point_at_distance(travelled)

        if jitter and self.noise_m > 0:
            lat, lon = offset_metres(
                (lat, lon),
                north_m=self._rng.gauss(0.0, self.noise_m),
                east_m=self._rng.gauss(0.0, self.noise_m),
            )

        # A small speed wobble makes the HUD look alive and feeds the ISI speed
        # term with slightly varying values.
        speed = max(0.0, self.speed_kmph + self._rng.gauss(0.0, 1.5))

        return GPSFix(
            timestamp_s=round(timestamp_s, 3),
            latitude=round(lat, 7),
            longitude=round(lon, 7),
            speed_kmph=round(speed, 1),
            heading_deg=round(heading, 1),
        )

    # --------------------------------------------------------------------------
    def describe(self) -> str:
        return (f"GPSTrack: {len(self.waypoints)} waypoints, "
                f"{self.total_length_m:.0f} m, "
                f"{self.speed_kmph:.0f} km/h, "
                f"~{self.duration_s:.0f} s to drive, "
                f"noise sigma {self.noise_m:.1f} m")


# ==============================================================================
# POTHOLE GEOLOCATION -- the step that makes clustering meaningful
# ==============================================================================
def project_pothole(fix: GPSFix,
                    distance_m: float,
                    lateral_m: float) -> Tuple[float, float]:
    """
    Convert a detection's *relative* position into an absolute coordinate.

    WHY THIS MATTERS
    ----------------
    (This is the single most important correction in the geospatial half of the
    pipeline, and the one most prototypes get wrong.)

    The naive implementation tags each detection with the car's own GPS fix.
    That is incorrect, and it fails in a way that quietly destroys clustering:

        A pothole is first detected ~25 m ahead and stays visible until the car
        is ~3 m away. If you log the CAR's position, those ~50 detections are
        smeared along 22 metres of road. DBSCAN at a 3 m radius then shreds one
        pothole into seven clusters -- while two genuinely distinct potholes
        10 m apart have their detection trails tangled together and merged.

    The fix is to push the fix forward by the measured range and sideways by the
    measured lateral offset, both rotated into the vehicle's heading:

        forward unit vector : ( cos H,  sin H)   in (north, east)
        right   unit vector : (-sin H,  cos H)

        north = distance * cos H  -  lateral * sin H
        east  = distance * sin H  +  lateral * cos H

    Now all 50 detections land on the same patch of road (within GPS noise),
    which is exactly the density signal DBSCAN is designed to find.

    Parameters
    ----------
    fix : the vehicle's GNSS fix at this instant.
    distance_m : range from camera to pothole, from the projection stage.
    lateral_m : sideways offset, +ve to the right of the vehicle centreline.

    Returns
    -------
    (latitude, longitude) of the pothole itself.
    """
    h = math.radians(fix.heading_deg)
    cos_h, sin_h = math.cos(h), math.sin(h)

    north = distance_m * cos_h - lateral_m * sin_h
    east = distance_m * sin_h + lateral_m * cos_h

    lat, lon = offset_metres((fix.latitude, fix.longitude), north, east)
    return round(lat, 7), round(lon, 7)


########################################################################
# END FILE: gps_simulator.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: renderer.py
# ==============================================================================

# BEGIN FILE: renderer.py
########################################################################

"""
================================================================================
 renderer.py  --  Annotation overlay for the output video
================================================================================

Draws, per frame:
  * A severity-coloured bounding box per pothole, with corner accents so the
    box stays readable against a busy road texture.
  * A label chip: confidence %, estimated depth, severity.
  * A driver HUD: speed, live GPS, running counts by severity.
  * A pulsing red warning banner whenever a High-severity pothole is in frame.

All drawing is pure OpenCV -- no extra dependencies, and it runs comfortably
faster than real time on a CPU.
================================================================================
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==============================================================================
# LOW-LEVEL HELPERS
# ==============================================================================
def _alpha_rect(img: np.ndarray,
                p1: Tuple[int, int],
                p2: Tuple[int, int],
                colour: Tuple[int, int, int],
                alpha: float) -> None:
    """Blend a filled rectangle onto `img` in place (used for chips/panels)."""
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full(roi.shape, colour, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _corner_box(img: np.ndarray,
                box: Tuple[int, int, int, int],
                colour: Tuple[int, int, int],
                thickness: int = 2,
                corner_frac: float = 0.28) -> None:
    """
    Draw a bounding box as four corner brackets plus a faint full outline.

    Corner brackets read much better than a solid rectangle when the background
    is high-frequency texture like asphalt.
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)

    cl = int(min(x2 - x1, y2 - y1) * corner_frac)
    cl = max(cl, 6)

    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * cl, py), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * cl), colour, thickness, cv2.LINE_AA)


# ==============================================================================
# DETECTION OVERLAY
# ==============================================================================
def draw_detection(frame: np.ndarray,
                   box: Tuple[float, float, float, float],
                   confidence: float,
                   assessment,
                   track_id: Optional[int] = None) -> None:
    """
    Annotate one detected pothole.

    Parameters
    ----------
    frame : the BGR image, modified in place.
    box : (x1, y1, x2, y2) pixel coordinates.
    confidence : detector confidence, 0..1.
    assessment : an ImpactAssessment from depth_impact.assess().
    track_id : optional persistent object ID.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    colour = config.SEVERITY_COLORS_BGR.get(assessment.severity, (200, 200, 200))

    # Severity tint inside the box -- subtle, so the road stays visible.
    _alpha_rect(frame, (x1, y1), (x2, y2), colour, 0.16)
    _corner_box(frame, (x1, y1, x2, y2), colour, thickness=2)

    # ---- label chip -------------------------------------------------------
    label = f"POTHOLE {confidence * 100:.0f}%"
    sub = f"{assessment.depth_cm:.1f}cm | {assessment.severity.upper()}"

    (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.48, 1)
    (sw, sh), _ = cv2.getTextSize(sub, _FONT, 0.42, 1)
    chip_w = max(lw, sw) + 14
    chip_h = lh + sh + 16

    # Prefer above the box; flip below if it would run off the top edge.
    cy1 = y1 - chip_h - 4
    if cy1 < 2:
        cy1 = min(y2 + 4, frame.shape[0] - chip_h - 2)
    cx1 = min(max(x1, 2), frame.shape[1] - chip_w - 2)

    _alpha_rect(frame, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), (18, 18, 18), 0.78)
    cv2.rectangle(frame, (cx1, cy1), (cx1 + 4, cy1 + chip_h), colour, -1)

    cv2.putText(frame, label, (cx1 + 10, cy1 + lh + 4),
                _FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, sub, (cx1 + 10, cy1 + lh + sh + 10),
                _FONT, 0.42, colour, 1, cv2.LINE_AA)

    # ---- distance readout, drawn under the box ---------------------------
    dist = f"{assessment.distance_m:.1f} m"
    if track_id is not None:
        dist += f"  #{track_id}"
    cv2.putText(frame, dist, (x1, min(y2 + 15, frame.shape[0] - 4)),
                _FONT, 0.40, colour, 1, cv2.LINE_AA)


# ==============================================================================
# DRIVER HUD
# ==============================================================================
def draw_hud(frame: np.ndarray,
             fix,
             counts: dict,
             frame_index: int,
             fps: float,
             detector_name: str) -> None:
    """Top-left status panel: speed, position, running detection tallies."""
    if not config.DRAW_HUD:
        return

    h, w = frame.shape[:2]
    pw, ph = 292, 116
    _alpha_rect(frame, (12, 12), (12 + pw, 12 + ph), (20, 20, 24), 0.72)

    cv2.putText(frame, "POTHOLE-AI  |  LIVE", (24, 36),
                _FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{config.VEHICLE_TYPE} - {detector_name}", (24, 55),
                _FONT, 0.40, (170, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fix.speed_kmph:.0f} km/h", (24, 79),
                _FONT, 0.50, (120, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fix.latitude:.5f}, {fix.longitude:.5f}", (110, 79),
                _FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Severity tallies, colour-matched to the boxes.
    x = 24
    for sev in ("Low", "Medium", "High"):
        col = config.SEVERITY_COLORS_BGR[sev]
        txt = f"{sev[0]}:{counts.get(sev, 0)}"
        cv2.circle(frame, (x + 4, 100), 4, col, -1, cv2.LINE_AA)
        cv2.putText(frame, txt, (x + 13, 104), _FONT, 0.42, col, 1, cv2.LINE_AA)
        x += 74

    # Timestamp, bottom-right.
    ts = frame_index / max(fps, 1e-6)
    stamp = f"t={ts:6.2f}s  f={frame_index}"
    (tw, _), _ = cv2.getTextSize(stamp, _FONT, 0.42, 1)
    cv2.putText(frame, stamp, (w - tw - 16, h - 14),
                _FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


# ==============================================================================
# WARNING BANNER
# ==============================================================================
def draw_warning(frame: np.ndarray,
                 frame_index: int,
                 message: str = "SEVERE POTHOLE AHEAD - SLOW DOWN") -> None:
    """
    Full-width red banner plus a pulsing border.

    The pulse is a sine of the frame index, so the alert visibly animates rather
    than sitting as a static block the eye stops noticing.
    """
    h, w = frame.shape[:2]
    pulse = 0.55 + 0.35 * abs(np.sin(frame_index * 0.32))

    # Border flash around the whole frame.
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                  (0, 0, int(255 * pulse)), 8)

    # Banner.
    by1, by2 = int(h * 0.80), int(h * 0.80) + 52
    _alpha_rect(frame, (0, by1), (w, by2), (0, 0, 190), pulse * 0.85)

    (tw, th), _ = cv2.getTextSize(message, _FONT, 0.82, 2)
    cv2.putText(frame, message, ((w - tw) // 2, by1 + (52 + th) // 2 - 2),
                _FONT, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

    # Warning triangle to the left of the text.
    tx, ty = (w - tw) // 2 - 42, by1 + 26
    pts = np.array([[tx, ty - 13], [tx - 14, ty + 11], [tx + 14, ty + 11]], np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.putText(frame, "!", (tx - 4, ty + 9), _FONT, 0.55, (0, 0, 190), 2, cv2.LINE_AA)


def draw_horizon(frame: np.ndarray) -> None:
    """Debug aid: draw the assumed horizon line used by the geometry model."""
    if not config.DRAW_HORIZON_LINE:
        return
    y = int(frame.shape[0] * config.HORIZON_RATIO)
    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "assumed horizon", (10, y - 6),
                _FONT, 0.38, (0, 255, 255), 1, cv2.LINE_AA)


########################################################################
# END FILE: renderer.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: video_utils.py
# ==============================================================================

# BEGIN FILE: video_utils.py
########################################################################

"""
================================================================================
 video_utils.py  --  Optional H.264 re-encoding
================================================================================

OpenCV's bundled FFmpeg build usually cannot write H.264 directly (the `avc1`
fourcc fails to open), so `cv2.VideoWriter` falls back to `mp4v` -- an MPEG-4
Part 2 codec that is universally readable but produces files roughly 5-10x
larger than H.264 and will not play inline in most browsers or in Google Slides.

If the `ffmpeg` binary is on PATH we transcode the finished file in a single
pass. If it is not, we leave the mp4v file exactly as it is and say so. Either
way the pipeline succeeds -- this is a nicety, not a dependency.
================================================================================
"""

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    """True if an `ffmpeg` binary can be found on PATH."""
    return shutil.which("ffmpeg") is not None


def compress_h264(path: Path,
                  crf: int = 24,
                  preset: str = "medium",
                  replace: bool = True) -> Path:
    """
    Re-encode `path` to H.264 (yuv420p) in place.

    Parameters
    ----------
    crf : Constant Rate Factor, 0 (lossless) to 51 (worst). 23-26 is visually
        transparent for dashcam footage.
    preset : encoder speed/efficiency trade-off.
    replace : if True the original file is overwritten with the compressed one.

    Returns
    -------
    Path to the resulting file (the original path if ffmpeg is unavailable).
    """
    path = Path(path)
    if not path.exists():
        return path

    if not ffmpeg_available():
        print("[video] ffmpeg not found -- keeping the mp4v file as-is.")
        print("        Install ffmpeg for a ~5x smaller, browser-playable file.")
        return path

    tmp = path.with_name(path.stem + "_h264.mp4")
    before_mb = path.stat().st_size / 1e6

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",       # required for QuickTime / browser playback
        "-movflags", "+faststart",   # metadata at the front, for streaming
        str(tmp),
    ]

    try:
        subprocess.run(cmd, check=True, timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[video] ffmpeg transcode failed ({exc}) -- keeping the original.")
        tmp.unlink(missing_ok=True)
        return path

    after_mb = tmp.stat().st_size / 1e6
    if replace:
        tmp.replace(path)
        result = path
    else:
        result = tmp

    print(f"[video] H.264 re-encode: {before_mb:.1f} MB -> {after_mb:.1f} MB "
          f"({100 * (1 - after_mb / max(before_mb, 1e-9)):.0f}% smaller)")
    return result


########################################################################
# END FILE: video_utils.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: detector.py
# ==============================================================================

# BEGIN FILE: detector.py
########################################################################

"""
================================================================================
 detector.py  --  Pluggable pothole detection backends
================================================================================

Two interchangeable backends behind one interface:

  YoloDetector       Real inference with ultralytics YOLOv8. Used whenever the
                     package is installed AND a trained weights file exists.

  SimulatedDetector  A deterministic, physics-plausible detection generator.
                     Used when ultralytics/weights are unavailable, or when
                     FORCE_SIMULATION is on.

WHY A SIMULATED BACKEND IS NOT CHEATING
---------------------------------------
The scientific contribution of this project is the *analysis* layer -- depth
proxy, ISI scoring, spatial clustering, map generation. Those stages must be
developed, unit-tested and demoed independently of whether a GPU and a labelled
dataset happen to be present. The simulator produces a detection stream with the
same statistical shape a real detector produces (persistent objects growing as
you approach them, confidence rising with proximity, occasional dropped frames,
occasional false positives), so every downstream stage is exercised honestly.

Both backends return the same object: a list of `Detection`.
================================================================================
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
# ==============================================================================
# COMMON INTERFACE
# ==============================================================================
@dataclass
class Detection:
    """One pothole detected in one frame, in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: Optional[int] = None   # stable ID across frames, when available

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class BaseDetector:
    """Interface every backend implements."""

    name = "base"

    def detect(self, frame, frame_index: int, timestamp_s: float) -> List[Detection]:
        raise NotImplementedError


# ==============================================================================
# BACKEND 1 -- REAL YOLOv8
# ==============================================================================
class YoloDetector(BaseDetector):
    """
    Thin wrapper around ultralytics YOLOv8.

    Uses `model.track()` rather than `model.predict()` so each pothole carries a
    persistent track_id across frames. That ID is not strictly required (DBSCAN
    can merge duplicates on geography alone) but it makes the CSV far easier to
    audit and lets us count how many frames each pothole was visible for.
    """

    name = "yolov8"

    def __init__(self, weights_path: Path):
        from ultralytics import YOLO           # imported lazily -- see load_detector

        self.model = YOLO(str(weights_path))
        self.weights_path = weights_path

        # Resolve which class indices count as "pothole". A model trained purely
        # on potholes has a single class; a road-damage model may have several.
        names = self.model.names
        wanted = {n.lower() for n in config.POTHOLE_CLASS_NAMES}
        self.target_ids = {
            idx for idx, nm in names.items() if str(nm).lower() in wanted
        }
        if not self.target_ids:            # single-class model -> accept everything
            self.target_ids = set(names.keys())

    # --------------------------------------------------------------------
    def detect(self, frame, frame_index: int, timestamp_s: float) -> List[Detection]:
        results = self.model.track(
            frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            max_det=config.MAX_DETECTIONS,
            persist=True,        # keep the tracker state between calls
            verbose=False,
        )

        out: List[Detection] = []
        if not results:
            return out

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else 0
            if cls_id not in self.target_ids:
                continue

            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            tid = int(boxes.id[i].item()) if boxes.id is not None else None

            out.append(Detection(x1, y1, x2, y2, conf, tid))
        return out


# ==============================================================================
# BACKEND 2 -- DETERMINISTIC SIMULATOR
# ==============================================================================
class SimulatedDetector(BaseDetector):
    """
    Generates a realistic detection stream without any neural network.

    MODEL
    -----
    Seed N potholes at fixed distances along the road. On every frame, advance
    the vehicle, and for each pothole still ahead of the camera:

      1. Project its real-world size and position through the same pinhole model
         the analysis stage uses -- so the boxes are geometrically consistent.
      2. Compute a confidence that grows as the pothole gets closer (a real
         detector is more certain about large, well-resolved objects).
      3. Drop the detection at random with probability DROP_RATE, mimicking the
         flicker every real detector exhibits.
      4. Occasionally emit a false positive (a tar patch or a shadow), so the
         clustering stage has some noise to be robust against.

    Everything is driven by a seeded RNG, so two runs produce identical output.
    """

    name = "simulated"

    DROP_RATE = 0.12               # per-frame probability of missing a real pothole
    FALSE_POSITIVE_RATE = 0.015    # per-frame probability of a spurious box
    VISIBLE_RANGE_M = 26.0         # how far ahead a pothole becomes detectable
    NEAR_CLIP_M = 3.2              # once this close it leaves the field of view

    def __init__(self,
                 frame_w: int,
                 frame_h: int,
                 speed_mps: float,
                 n_potholes: int = config.SAMPLE_POTHOLE_COUNT,
                 duration_s: float = 20.0,
                 seed: int = config.GPS_RANDOM_SEED,
                 groundtruth_path: Optional[Path] = None):
        self.w, self.h = frame_w, frame_h
        self.speed_mps = speed_mps
        self._rng = random.Random(seed)

        # Same camera model as depth_impact.GroundPlaneProjector -- kept in sync
        # so simulated boxes decode back to the sizes we seeded them with.
        half_fov = math.radians(config.CAMERA_HFOV_DEG) / 2.0
        self.focal_px = (frame_w / 2.0) / math.tan(half_fov)
        self.horizon_y = frame_h * config.HORIZON_RATIO

        # ---- source the pothole layout -----------------------------------
        # Preferred: the ground-truth sidecar written by make_sample_video.py.
        # Using it means the simulated boxes land exactly on the potholes that
        # were actually painted into the frames, so the annotated video is
        # visually coherent rather than boxes floating over clean asphalt.
        loaded = self._load_groundtruth(groundtruth_path)
        if loaded is not None:
            self.potholes = loaded
            self.source = "groundtruth"
            return

        # Fallback: synthesise a fresh layout (used with third-party footage).
        self.source = "synthetic"
        road_length_m = speed_mps * duration_s + self.VISIBLE_RANGE_M
        self.potholes = []
        for pid in range(n_potholes):
            self.potholes.append({
                "id": pid + 1,
                # Spread them out, but not perfectly evenly.
                "s_m": (pid + 0.6) * road_length_m / (n_potholes + 1)
                       + self._rng.uniform(-4.0, 4.0),
                # Lateral offset from the camera's optical axis (metres).
                "lateral_m": self._rng.uniform(-1.6, 1.6),
                # Real-world footprint. The spread deliberately covers all three
                # severity bands so the demo map is colourful and informative.
                "width_m": self._rng.uniform(0.28, 1.45),
                "length_m": self._rng.uniform(0.25, 1.30),
                "quality": self._rng.uniform(0.72, 0.97),   # peak confidence
            })

    # --------------------------------------------------------------------
    def _load_groundtruth(self, path: Optional[Path]) -> Optional[list]:
        """Read the sidecar JSON emitted next to a generated sample video."""
        candidates = [path] if path else []
        candidates.append(config.INPUT_VIDEO.with_suffix(".groundtruth.json"))

        for cand in candidates:
            if cand is None or not Path(cand).exists():
                continue
            try:
                blob = json.loads(Path(cand).read_text(encoding="utf-8"))
                pots = []
                for i, p in enumerate(blob["potholes"]):
                    pots.append({
                        "id": i + 1,
                        "s_m": float(p["s_m"]),
                        "lateral_m": float(p["lateral_m"]),
                        "width_m": float(p["width_m"]),
                        "length_m": float(p["length_m"]),
                        "quality": 0.72 + 0.25 * self._rng.random(),
                    })
                print(f"[detector] Simulated boxes aligned to ground truth "
                      f"({len(pots)} potholes from {Path(cand).name})")
                return pots
            except Exception:                       # noqa: BLE001
                continue
        return None

    # --------------------------------------------------------------------
    def _project(self, distance_m: float, lateral_m: float,
                 width_m: float, length_m: float
                 ) -> Optional[Tuple[float, float, float, float]]:
        """Invert the ground-plane model: real-world pothole -> pixel box."""
        if distance_m <= self.NEAR_CLIP_M:
            return None

        # Row for the near and far edges of the pothole.
        y_near = self.horizon_y + (self.focal_px * config.CAMERA_HEIGHT_M) / distance_m
        far = distance_m + length_m
        y_far = self.horizon_y + (self.focal_px * config.CAMERA_HEIGHT_M) / far

        # Horizontal extent at the near edge.
        cx = self.w / 2.0 + (lateral_m * self.focal_px) / distance_m
        half_w_px = (width_m / 2.0) * self.focal_px / distance_m

        x1, x2 = cx - half_w_px, cx + half_w_px
        y1, y2 = y_far, y_near

        # Reject anything essentially off-screen or sub-pixel.
        if x2 < 4 or x1 > self.w - 4 or (x2 - x1) < 8 or (y2 - y1) < 5:
            return None
        if y2 > self.h - 2:
            return None

        # Clip to the frame.
        return (max(0.0, x1), max(0.0, y1),
                min(float(self.w), x2), min(float(self.h), y2))

    # --------------------------------------------------------------------
    def detect(self, frame, frame_index: int, timestamp_s: float) -> List[Detection]:
        vehicle_s = self.speed_mps * timestamp_s
        out: List[Detection] = []

        for ph in self.potholes:
            distance = ph["s_m"] - vehicle_s
            if not (self.NEAR_CLIP_M < distance < self.VISIBLE_RANGE_M):
                continue

            box = self._project(distance, ph["lateral_m"],
                                ph["width_m"], ph["length_m"])
            if box is None:
                continue

            # Confidence rises as the object gets closer / bigger.
            proximity = 1.0 - (distance - self.NEAR_CLIP_M) / self.VISIBLE_RANGE_M
            conf = ph["quality"] * (0.45 + 0.55 * proximity)
            conf = min(0.98, max(0.0, conf + self._rng.gauss(0, 0.02)))

            if conf < config.CONF_THRESHOLD:
                continue
            if self._rng.random() < self.DROP_RATE:      # detector flicker
                continue

            # Small per-frame box jitter, exactly like a real detector.
            jx = self._rng.gauss(0, 1.6)
            jy = self._rng.gauss(0, 1.2)
            out.append(Detection(box[0] + jx, box[1] + jy,
                                 box[2] + jx, box[3] + jy,
                                 round(conf, 3), ph["id"]))

        # ---- occasional false positive -----------------------------------
        if self._rng.random() < self.FALSE_POSITIVE_RATE:
            fx = self._rng.uniform(0.15, 0.85) * self.w
            fy = self._rng.uniform(0.62, 0.88) * self.h
            fw = self._rng.uniform(30, 70)
            fh = fw * self._rng.uniform(0.35, 0.6)
            out.append(Detection(fx, fy, fx + fw, fy + fh,
                                 round(self._rng.uniform(0.36, 0.52), 3),
                                 track_id=9000 + frame_index % 50))

        return out[:config.MAX_DETECTIONS]


# ==============================================================================
# FACTORY -- graceful dependency handling
# ==============================================================================
def load_detector(frame_w: int,
                  frame_h: int,
                  speed_mps: float,
                  duration_s: float) -> BaseDetector:
    """
    Return the best available detector, never raising.

    Decision order:
        1. FORCE_SIMULATION set             -> SimulatedDetector
        2. ultralytics missing              -> SimulatedDetector (with a notice)
        3. weights file missing             -> SimulatedDetector (with a notice)
        4. otherwise                        -> YoloDetector
    """
    if config.FORCE_SIMULATION:
        print("[detector] FORCE_SIMULATION is on -> using SimulatedDetector.")
        return SimulatedDetector(frame_w, frame_h, speed_mps, duration_s=duration_s)

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("[detector] `ultralytics` not installed -> using SimulatedDetector.")
        print("           Install with:  pip install ultralytics")
        return SimulatedDetector(frame_w, frame_h, speed_mps, duration_s=duration_s)

    if not config.YOLO_WEIGHTS.exists():
        print(f"[detector] No weights at {config.YOLO_WEIGHTS} -> SimulatedDetector.")
        print("           Train your own with:  python train_yolo.py")
        return SimulatedDetector(frame_w, frame_h, speed_mps, duration_s=duration_s)

    try:
        det = YoloDetector(config.YOLO_WEIGHTS)
        print(f"[detector] Loaded YOLOv8 weights: {config.YOLO_WEIGHTS.name}")
        return det
    except Exception as exc:                                # noqa: BLE001
        print(f"[detector] YOLO failed to load ({exc}) -> SimulatedDetector.")
        return SimulatedDetector(frame_w, frame_h, speed_mps, duration_s=duration_s)


########################################################################
# END FILE: detector.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: clustering.py
# ==============================================================================

"""
================================================================================
 clustering.py  --  DBSCAN spatial de-duplication of raw detections
================================================================================

THE PROBLEM
-----------
The raw CSV is a *per-frame* log. One physical pothole approached at 34 km/h is
visible for roughly 2 seconds -- about 45 frames -- so it contributes ~40 rows,
each with a slightly different GPS fix because of GNSS jitter. Plot that raw log
and you get a smeared blob, not a map. Worse, a road maintenance crew reading it
would think there were 40 potholes.

THE SOLUTION
------------
Density-based clustering on the geographic coordinates. DBSCAN is the right
algorithm here for three specific reasons:

  1. It does NOT need `k` specified in advance. We do not know how many distinct
     potholes a drive will find -- that is the entire question being asked.
  2. Its `eps` parameter is a *physical distance*, which maps directly onto our
     requirement: "merge everything within 3 metres."
  3. It has a native concept of noise, so isolated one-frame false positives can
     be quarantined rather than promoted to map markers.

THE HAVERSINE DETAIL (the part people get wrong)
------------------------------------------------
You cannot run DBSCAN on raw lat/lon with Euclidean distance. One degree of
longitude is ~111 km at the equator but ~0 km at the poles, so a fixed eps means
a different physical radius depending on where you are. scikit-learn supports
`metric='haversine'`, which computes true great-circle distance -- but it
requires the coordinates in RADIANS and returns distances in radians of arc.
So eps must be converted:

        eps_radians = radius_in_metres / EARTH_RADIUS_M

That single line is the difference between a correct cluster and a cluster that
silently mis-merges the moment you test in a different city.
================================================================================
"""

from typing import Optional

import numpy as np
import pandas as pd
# ==============================================================================
# CORE CLUSTERING
# ==============================================================================
def cluster_detections(df: pd.DataFrame,
                       radius_m: float = config.CLUSTER_RADIUS_M,
                       min_samples: int = config.CLUSTER_MIN_SAMPLES
                       ) -> pd.DataFrame:
    """
    Assign a `cluster_id` to every row of the raw detection log.

    Parameters
    ----------
    df : DataFrame with at least `latitude` and `longitude` columns.
    radius_m : merge radius in metres (DBSCAN's eps).
    min_samples : minimum points to form a dense region. Set to 1 so that a
        genuine single-frame detection still becomes its own cluster rather than
        being discarded as noise -- we filter low-confidence clusters later using
        the more meaningful `hit_count` instead.

    Returns
    -------
    The same DataFrame with an added integer `cluster_id` column (-1 = noise).
    """
    if df.empty:
        return df.assign(cluster_id=pd.Series(dtype="int64"))

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for clustering. Install with:\n"
            "    pip install scikit-learn"
        ) from exc

    # ---- 1. coordinates -> radians ---------------------------------------
    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))

    # ---- 2. metres -> radians of arc -------------------------------------
    eps_rad = radius_m / config.EARTH_RADIUS_M

    # ---- 3. run DBSCAN ----------------------------------------------------
    # ball_tree is the only algorithm in sklearn that supports haversine.
    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = model.fit_predict(coords_rad)

    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    return out


# ==============================================================================
# AGGREGATION -- many detections -> one verified map marker
# ==============================================================================
def summarise_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a clustered detection log into one row per physical pothole.

    Aggregation choices, and why:

    latitude / longitude   MEAN of the members. Averaging N independent GNSS
                           fixes reduces the position error by sqrt(N) -- a real
                           accuracy win, not just tidiness.

    depth_cm / area_m2     MEDIAN, not mean. The median is robust to the one
                           frame where the box blew up because of a shadow.

    confidence             MAX. The best look the detector got at this pothole,
                           usually the closest frame.

    severity               The WORST severity observed. Under-reporting a
                           dangerous pothole is far costlier than over-reporting
                           a mild one, so this stage is deliberately pessimistic.

    hit_count              How many frames saw it. This is the trust signal:
                           potholes seen once are probably false positives,
                           potholes seen 30 times are certain.

    Returns
    -------
    DataFrame, one row per cluster, sorted by severity then hit_count.
    """
    if df.empty or "cluster_id" not in df.columns:
        return pd.DataFrame(columns=[
            "cluster_id", "latitude", "longitude", "depth_cm", "area_m2",
            "confidence", "severity", "isi", "hit_count", "verified",
            "first_seen_s", "last_seen_s", "vehicle",
        ])

    # Drop DBSCAN noise points (only possible when min_samples > 1).
    valid = df[df["cluster_id"] >= 0].copy()
    if valid.empty:
        return summarise_clusters(pd.DataFrame())

    

    grouped = valid.groupby("cluster_id", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        # 90th percentile, not max: pessimistic about the pothole but robust to
        # the one frame where the detector's box blew up on a shadow.
        depth_cm=("depth_cm", lambda s: s.quantile(0.90)),
        area_m2=("area_m2", lambda s: s.quantile(0.90)),
        isi=("isi", lambda s: s.quantile(0.90)),
        depth_median_cm=("depth_cm", "median"),
        confidence=("confidence", "max"),
        hit_count=("frame_index", "count"),
        first_seen_s=("timestamp_s", "min"),
        last_seen_s=("timestamp_s", "max"),
        vehicle=("vehicle", "first"),
    )

    # Derive the label FROM the aggregated index, never by voting on the
    # per-frame labels. Voting lets a cluster report severity "High" alongside
    # an ISI of 0.30, which is incoherent to anyone reading the CSV.
    grouped["severity"] = grouped["isi"].apply(classify_severity)

    # A cluster becomes "verified" once enough independent frames agree.
    grouped["verified"] = grouped["hit_count"] >= config.VERIFIED_MIN_HITS

    # Round for a human-readable CSV.
    for col, nd in [("latitude", 7), ("longitude", 7), ("depth_cm", 2),
                    ("depth_median_cm", 2), ("area_m2", 4), ("isi", 4),
                    ("confidence", 3), ("first_seen_s", 2), ("last_seen_s", 2)]:
        grouped[col] = grouped[col].round(nd)

    severity_rank = {"Low": 0, "Medium": 1, "High": 2}
    grouped["_sort"] = grouped["severity"].map(severity_rank)
    grouped = (grouped.sort_values(["_sort", "hit_count"], ascending=[False, False])
                      .drop(columns=["_sort"])
                      .reset_index(drop=True))
    return grouped


# ==============================================================================
# CONVENIENCE WRAPPER -- CSV in, CSV out
# ==============================================================================
def cluster_csv(input_csv=config.LOG_CSV,
                output_csv=config.CLUSTER_CSV,
                radius_m: float = config.CLUSTER_RADIUS_M
                ) -> Optional[pd.DataFrame]:
    """
    Read the raw log, cluster it, write the summary, and return it.

    This is the function to call if you want to re-cluster an existing run with
    a different radius without re-processing the video:

        >>> from src.clustering import cluster_csv
        >>> cluster_csv(radius_m=5.0)
    """
    import pathlib

    input_csv = pathlib.Path(input_csv)
    if not input_csv.exists():
        print(f"[cluster] Input log not found: {input_csv}")
        return None

    raw = pd.read_csv(input_csv)
    if raw.empty:
        print("[cluster] Raw log is empty -- nothing to cluster.")
        return None

    labelled = cluster_detections(raw, radius_m=radius_m)
    summary = summarise_clusters(labelled)

    summary.to_csv(output_csv, index=False)

    n_raw, n_clusters = len(labelled), len(summary)
    reduction = (1 - n_clusters / n_raw) * 100 if n_raw else 0.0
    print(f"[cluster] {n_raw} raw detections -> {n_clusters} unique potholes "
          f"({reduction:.1f}% de-duplicated) @ {radius_m:.1f} m radius")
    print(f"[cluster] Wrote {output_csv}")
    return summary

# ==============================================================================
# EMBEDDED MODULE: mapping.py
# ==============================================================================

# BEGIN FILE: mapping.py
########################################################################

"""
================================================================================
 mapping.py  --  Interactive folium map of clustered potholes
================================================================================

Turns the clustered CSV into a single self-contained HTML file that a municipal
engineer can open in any browser, with no server and no install.

Layers produced
---------------
  * Driven route polyline (context: where did the survey actually go)
  * One circle marker per verified pothole, colour-coded by severity
      Green  = Low     Orange = Medium     Red = High
  * Marker radius scales with estimated depth, so severity is encoded twice
    (colour + size) -- important for colour-blind accessibility
  * Rich popup per marker: depth, area, ISI, confidence, hit count, coordinates
  * A heat map layer showing damage density along the corridor
  * A layer switcher so any of the above can be toggled
  * A fixed HTML legend, injected directly into the map's root
================================================================================
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
# ==============================================================================
# LEGEND -- injected as raw HTML into the folium root element
# ==============================================================================
_LEGEND_TEMPLATE = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed; bottom: 28px; left: 18px; z-index: 9999;
    background: rgba(255,255,255,0.95); padding: 14px 16px;
    border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color:#222;">
  <div style="font-weight:700; font-size:14px; margin-bottom:8px;">
    Impact Severity &mdash; __VEHICLE__
  </div>
  <div style="margin:4px 0;">
    <span style="display:inline-block;width:13px;height:13px;border-radius:50%;
                 background:#d63a2f;margin-right:8px;"></span>High &mdash; avoid / repair urgently
  </div>
  <div style="margin:4px 0;">
    <span style="display:inline-block;width:13px;height:13px;border-radius:50%;
                 background:#f0932b;margin-right:8px;"></span>Medium &mdash; slow down
  </div>
  <div style="margin:4px 0;">
    <span style="display:inline-block;width:13px;height:13px;border-radius:50%;
                 background:#3aa655;margin-right:8px;"></span>Low &mdash; monitor
  </div>
  <hr style="margin:9px 0; border:none; border-top:1px solid #ddd;">
  <div style="font-size:11.5px;color:#555;">
    Marker size &prop; estimated depth<br>
    __SUMMARY__
  </div>
</div>
{% endmacro %}
"""

_HEX = {"Low": "#3aa655", "Medium": "#f0932b", "High": "#d63a2f"}


# ==============================================================================
def _popup_html(row: pd.Series) -> str:
    """Build the HTML card shown when a marker is clicked."""
    colour = _HEX.get(row["severity"], "#555")
    badge = "VERIFIED" if row.get("verified", False) else "UNCONFIRMED"
    badge_bg = "#2d7d46" if row.get("verified", False) else "#8a8a8a"

    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;width:248px;">
      <div style="background:{colour};color:#fff;padding:8px 11px;
                  border-radius:6px 6px 0 0;font-weight:700;font-size:14px;">
        {row['severity']} Impact &nbsp;&middot;&nbsp; Pothole #{int(row['cluster_id'])}
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:12.5px;">
        <tr><td style="padding:5px 10px;color:#666;">Est. depth</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {row['depth_cm']:.1f} cm</td></tr>
        <tr style="background:#f7f7f7;"><td style="padding:5px 10px;color:#666;">Surface area</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {row['area_m2']:.2f} m&sup2;</td></tr>
        <tr><td style="padding:5px 10px;color:#666;">Impact index</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {row['isi']:.3f}</td></tr>
        <tr style="background:#f7f7f7;"><td style="padding:5px 10px;color:#666;">Detector conf.</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {row['confidence']*100:.0f}%</td></tr>
        <tr><td style="padding:5px 10px;color:#666;">Frames seen</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {int(row['hit_count'])}</td></tr>
        <tr style="background:#f7f7f7;"><td style="padding:5px 10px;color:#666;">Reference vehicle</td>
            <td style="padding:5px 10px;text-align:right;font-weight:600;">
                {row.get('vehicle','-')}</td></tr>
      </table>
      <div style="padding:7px 10px;border-top:1px solid #eee;font-size:11px;color:#777;">
        {row['latitude']:.6f}, {row['longitude']:.6f}
        <span style="float:right;background:{badge_bg};color:#fff;padding:1px 6px;
                     border-radius:3px;font-size:10px;">{badge}</span>
      </div>
    </div>
    """


# ==============================================================================
def build_map(clusters: pd.DataFrame,
              route: Optional[Sequence[Tuple[float, float]]] = None,
              output_html: Path = config.MAP_HTML,
              show_unverified: bool = True) -> Optional[Path]:
    """
    Render the interactive map.

    Parameters
    ----------
    clusters : output of clustering.summarise_clusters().
    route : optional list of (lat, lon) waypoints to draw as the survey path.
    output_html : destination file.
    show_unverified : if False, clusters below VERIFIED_MIN_HITS are hidden.

    Returns
    -------
    Path to the written HTML, or None if folium is unavailable / no data.
    """
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
        from branca.element import MacroElement, Template
    except ImportError:
        print("[map] `folium` not installed -- skipping map generation.")
        print("      Install with:  pip install folium")
        return None

    if clusters is None or clusters.empty:
        print("[map] No clusters to plot.")
        return None

    data = clusters if show_unverified else clusters[clusters["verified"]]
    if data.empty:
        print("[map] Every cluster was filtered out; nothing to plot.")
        return None

    # ---- base map centred on the detections -------------------------------
    centre = [data["latitude"].mean(), data["longitude"].mean()]
    fmap = folium.Map(location=centre, zoom_start=17, tiles=None, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="Street map").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False, control=True,
    ).add_to(fmap)

    # ---- survey route ------------------------------------------------------
    if route:
        folium.PolyLine(
            locations=[list(p) for p in route],
            color="#2c3e50", weight=4, opacity=0.55,
            tooltip="Surveyed route",
        ).add_to(fmap)

    # ---- one feature group per severity, so each can be toggled -----------
    groups = {}
    for sev in ("High", "Medium", "Low"):
        n = int((data["severity"] == sev).sum())
        groups[sev] = folium.FeatureGroup(name=f"{sev} impact ({n})", show=True)

    # Depth -> radius scaling. Clamped so tiny potholes stay clickable and
    # huge ones do not swallow the map.
    d_min, d_max = data["depth_cm"].min(), data["depth_cm"].max()
    span = max(d_max - d_min, 1e-6)

    for _, row in data.iterrows():
        norm = (row["depth_cm"] - d_min) / span
        radius = 7 + 13 * norm

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=_HEX.get(row["severity"], "#555"),
            fill_opacity=0.85 if row.get("verified", True) else 0.45,
            popup=folium.Popup(_popup_html(row), max_width=280),
            tooltip=f"{row['severity']} | {row['depth_cm']:.1f} cm deep",
        ).add_to(groups[row["severity"]])

    for g in groups.values():
        g.add_to(fmap)

    # ---- heat map of damage density ---------------------------------------
    weights = {"Low": 0.35, "Medium": 0.7, "High": 1.0}
    heat_points = [
        [r["latitude"], r["longitude"], weights.get(r["severity"], 0.5)]
        for _, r in data.iterrows()
    ]
    heat_layer = folium.FeatureGroup(name="Damage density (heat map)", show=False)
    HeatMap(heat_points, radius=26, blur=18, min_opacity=0.25).add_to(heat_layer)
    heat_layer.add_to(fmap)

    # ---- legend ------------------------------------------------------------
    counts = data["severity"].value_counts().to_dict()
    summary = (f"{len(data)} unique potholes &middot; "
               f"{counts.get('High', 0)} high &middot; "
               f"{counts.get('Medium', 0)} medium &middot; "
               f"{counts.get('Low', 0)} low")

    legend = MacroElement()
    legend._template = Template(
        _LEGEND_TEMPLATE
        .replace("__VEHICLE__", str(data["vehicle"].iloc[0]))
        .replace("__SUMMARY__", summary)
    )
    fmap.get_root().add_child(legend)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # ---- fit the viewport to the data -------------------------------------
    fmap.fit_bounds([
        [data["latitude"].min(), data["longitude"].min()],
        [data["latitude"].max(), data["longitude"].max()],
    ], padding=(40, 40))

    output_html = Path(output_html)
    fmap.save(str(output_html))
    print(f"[map] Wrote interactive map -> {output_html}")
    return output_html


# ==============================================================================
def map_from_csv(cluster_csv=config.CLUSTER_CSV,
                 output_html=config.MAP_HTML) -> Optional[Path]:
    """Convenience: build a map straight from an existing cluster CSV."""
    cluster_csv = Path(cluster_csv)
    if not cluster_csv.exists():
        print(f"[map] Cluster CSV not found: {cluster_csv}")
        return None
    return build_map(pd.read_csv(cluster_csv),
                     route=config.ROUTE_WAYPOINTS,
                     output_html=output_html)


########################################################################
# END FILE: mapping.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: make_sample_video.py
# ==============================================================================

"""
================================================================================
 make_sample_video.py  --  Procedural dashcam clip generator
================================================================================

PURPOSE
-------
A demo must never fail because someone did not download a video file. This
script renders `data/input_dashcam.mp4` from scratch with OpenCV: a road
surface, lane markings, a horizon, moving scenery, and potholes that approach
the camera with correct perspective.

The potholes are placed using the SAME pinhole camera model that
`src/depth_impact.py` uses to measure them. That is not a coincidence -- it is
what makes the synthetic clip a genuine end-to-end test: we seed a pothole of a
known real-world size, and the analysis stage should recover approximately that
size from the rendered pixels.

RUN
---
    python make_sample_video.py
    python make_sample_video.py --seconds 30 --potholes 20
================================================================================
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
# ==============================================================================
class RoadRenderer:
    """Renders a first-person view of a straight road with potholes."""

    def __init__(self, width: int, height: int, seed: int = 7):
        self.w, self.h = width, height
        self.rng = random.Random(seed)

        # Camera model, identical to the analysis stage.
        half_fov = math.radians(config.CAMERA_HFOV_DEG) / 2.0
        self.focal_px = (width / 2.0) / math.tan(half_fov)
        self.horizon_y = height * config.HORIZON_RATIO

        # A fixed asphalt noise texture, scrolled each frame to fake motion.
        self.noise = self.rng.Random if False else None
        rs = np.random.RandomState(seed)
        self.asphalt = rs.normal(0, 9, (height * 2, width)).astype(np.float32)

        self._build_backdrop()

    # ------------------------------------------------------------------
    def _build_backdrop(self) -> None:
        """Sky gradient + a static skyline, drawn once and reused."""
        bg = np.zeros((self.h, self.w, 3), np.uint8)

        hy = int(self.horizon_y)
        # Sky: cool blue at the top fading to hazy near the horizon.
        for y in range(hy):
            t = y / max(hy, 1)
            bg[y, :] = (int(205 - 35 * t), int(170 - 25 * t), int(135 - 20 * t))

        # Distant buildings, so the horizon is not a bare line.
        x = 0
        while x < self.w:
            bw = self.rng.randint(28, 74)
            bh = self.rng.randint(16, 62)
            shade = self.rng.randint(95, 130)
            cv2.rectangle(bg, (x, hy - bh), (x + bw, hy), (shade, shade - 6, shade - 12), -1)
            x += bw + self.rng.randint(3, 16)

        self.backdrop = bg

    # ------------------------------------------------------------------
    def _project_point(self, distance_m: float, lateral_m: float):
        """(distance ahead, lateral offset) -> pixel (x, y)."""
        if distance_m <= 0.4:
            return None
        y = self.horizon_y + (self.focal_px * config.CAMERA_HEIGHT_M) / distance_m
        x = self.w / 2.0 + (lateral_m * self.focal_px) / distance_m
        return x, y

    # ------------------------------------------------------------------
    def _draw_road(self, frame: np.ndarray, travelled_m: float) -> None:
        """Roadside terrain, asphalt trapezoid, kerbs and dashed lane markings."""
        hy = int(self.horizon_y)
        half_width_m = 4.6           # carriageway half-width

        # ---- 1. roadside terrain fills everything below the horizon -------
        # Drawn first, then the road polygon is painted on top. Without this the
        # asphalt would span the full frame width and the scene would read as a
        # runway rather than a road.
        cv2.rectangle(frame, (0, hy), (self.w, self.h), (86, 104, 96), -1)

        # ---- 2. asphalt trapezoid ------------------------------------------
        near_l = self._project_point(3.2, -half_width_m)
        near_r = self._project_point(3.2, half_width_m)
        far_l = self._project_point(120.0, -half_width_m)
        far_r = self._project_point(120.0, half_width_m)
        if not all((near_l, near_r, far_l, far_r)):
            return

        # Extend the near edge past the bottom of the frame so the road does not
        # float above the frame border.
        poly = np.array([
            [int(near_l[0]) - 260, self.h],
            [int(far_l[0]), int(far_l[1])],
            [int(far_r[0]), int(far_r[1])],
            [int(near_r[0]) + 260, self.h],
        ], np.int32)

        # Build the asphalt as a textured patch, then stencil it via the polygon.
        offset = int(travelled_m * 22) % self.h
        grain = self.asphalt[offset:offset + (self.h - hy), :]
        patch = np.clip(74.0 + grain, 30, 190).astype(np.uint8)
        asphalt = np.dstack([patch, patch, patch])

        mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        sub_mask = mask[hy:, :]
        frame[hy:, :][sub_mask > 0] = asphalt[sub_mask > 0]

        # ---- 3. kerb lines --------------------------------------------------
        for side in (-1, 1):
            near = self._project_point(3.4, side * half_width_m)
            far = self._project_point(110.0, side * half_width_m)
            if near and far:
                cv2.line(frame, (int(near[0]), int(near[1])),
                         (int(far[0]), int(far[1])), (176, 176, 172), 2, cv2.LINE_AA)

        # ---- dashed centre line -------------------------------------------
        # Dashes are anchored in world space (every 6 m) and scrolled by the
        # distance travelled, which produces correct perspective motion.
        dash_period, dash_len = 6.0, 2.6
        start = math.floor(travelled_m / dash_period) * dash_period
        for k in range(26):
            d0 = start + k * dash_period - travelled_m
            d1 = d0 + dash_len
            if d1 < 4.0:
                continue
            p0 = self._project_point(max(d0, 4.0), 0.0)
            p1 = self._project_point(max(d1, 4.2), 0.0)
            if not (p0 and p1):
                continue
            thick = max(1, int(26.0 / max(d0, 4.0)))
            cv2.line(frame, (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])), (225, 225, 220), thick, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_pothole(self, frame: np.ndarray, ph: dict, distance_m: float) -> None:
        """
        Render one pothole as a dark, irregular ellipse with a lighter rim.

        The rim (a bright crescent on the near edge) is what a real pothole looks
        like: the broken asphalt lip catches the light while the cavity is in
        shadow. It is also the visual cue a CNN learns.
        """
        if distance_m <= 3.0 or distance_m > 60.0:
            return

        near = self._project_point(distance_m, ph["lateral_m"])
        far = self._project_point(distance_m + ph["length_m"], ph["lateral_m"])
        if not (near and far):
            return

        cx = (near[0] + far[0]) / 2.0
        cy = (near[1] + far[1]) / 2.0
        ax = max(2, int((ph["width_m"] / 2.0) * self.focal_px / distance_m))
        ay = max(1, int(abs(near[1] - far[1]) / 2.0))

        if cy > self.h + 20 or cy < self.horizon_y:
            return

        c = (int(cx), int(cy))

        # Cavity: dark ellipse, softened by a blur pass over its ROI.
        cv2.ellipse(frame, c, (ax, ay), ph["angle"], 0, 360, (28, 26, 25), -1, cv2.LINE_AA)

        # Inner shadow gradient.
        cv2.ellipse(frame, (c[0], c[1] + max(1, ay // 4)),
                    (max(1, int(ax * 0.72)), max(1, int(ay * 0.6))),
                    ph["angle"], 0, 360, (16, 15, 15), -1, cv2.LINE_AA)

        # Broken rim on the near (lower) edge.
        cv2.ellipse(frame, c, (ax, ay), ph["angle"], 15, 165,
                    (112, 110, 106), max(1, ax // 12 + 1), cv2.LINE_AA)

        # Ragged edge detail -- a few random notches around the perimeter.
        for k in range(ph["notches"]):
            t = 2 * math.pi * k / max(ph["notches"], 1) + ph["seedphase"]
            nx = int(cx + ax * 1.02 * math.cos(t))
            ny = int(cy + ay * 1.02 * math.sin(t))
            r = max(1, int(ax * 0.16))
            cv2.circle(frame, (nx, ny), r, (34, 32, 31), -1, cv2.LINE_AA)

        # Local blur so the pothole is not razor-sharp against the grain.
        x0, x1 = max(0, c[0] - ax - 6), min(self.w, c[0] + ax + 6)
        y0, y1 = max(0, c[1] - ay - 6), min(self.h, c[1] + ay + 6)
        if x1 - x0 > 4 and y1 - y0 > 4:
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (3, 3), 0)

    # ------------------------------------------------------------------
    def render(self, potholes: list, travelled_m: float, frame_idx: int) -> np.ndarray:
        frame = self.backdrop.copy()
        self._draw_road(frame, travelled_m)

        # Far-to-near ordering so nearer potholes overdraw farther ones.
        visible = sorted(
            ((p, p["s_m"] - travelled_m) for p in potholes),
            key=lambda t: -t[1],
        )
        for ph, dist in visible:
            self._draw_pothole(frame, ph, dist)

        # Light sensor noise for a dashcam look. Kept low deliberately: heavy
        # per-frame noise is incompressible and inflates the mp4 enormously.
        noise = np.random.RandomState(frame_idx).normal(0, 1.1, (self.h, self.w, 1))
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Timestamp burn-in, as most dashcams do.
        cv2.putText(frame, f"DASHCAM  FRAME {frame_idx:05d}",
                    (self.w - 260, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (235, 235, 235), 1, cv2.LINE_AA)
        return frame


# ==============================================================================
def generate(output_path: Path = config.INPUT_VIDEO,
             seconds: int = config.SAMPLE_VIDEO_SECONDS,
             fps: int = config.SAMPLE_VIDEO_FPS,
             width: int = config.SAMPLE_VIDEO_WIDTH,
             height: int = config.SAMPLE_VIDEO_HEIGHT,
             n_potholes: int = config.SAMPLE_POTHOLE_COUNT,
             speed_kmph: float = config.VEHICLE_SPEED_KMPH,
             seed: int = 7) -> Path:
    """Render and write the synthetic dashcam clip."""
    rng = random.Random(seed)
    speed_mps = speed_kmph / 3.6
    total_frames = int(seconds * fps)
    road_len = speed_mps * seconds + 40.0

    # Seed potholes with real-world dimensions spanning all severity bands.
    potholes = []
    for i in range(n_potholes):
        potholes.append({
            # Jitter is kept below half the nominal spacing so no two potholes
            # are ever generated closer than the DBSCAN merge radius -- that
            # would make "correct" clustering genuinely ambiguous.
            "s_m": (i + 0.7) * road_len / (n_potholes + 1) + rng.uniform(-1.2, 1.2),
            "lateral_m": rng.uniform(-1.7, 1.7),
            "width_m": rng.uniform(0.30, 1.95),
            "length_m": rng.uniform(0.28, 1.75),
            "angle": rng.randint(0, 179),
            "notches": rng.randint(4, 9),
            "seedphase": rng.uniform(0, 6.28),
        })

    renderer = RoadRenderer(width, height, seed=seed)

    # ---- write ground truth alongside the video --------------------------
    # This sidecar serves two purposes:
    #   1. SimulatedDetector loads it so its boxes land on the potholes that
    #      were actually painted, instead of a second, unrelated random layout.
    #   2. It is a genuine ground-truth file: real dimensions of every pothole,
    #      which lets us measure how accurately the depth/area stage recovers
    #      them from pixels alone (see evaluate.py).
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path = output_path.with_suffix(".groundtruth.json")
    gt_path.write_text(json.dumps({
        "video": output_path.name,
        "fps": fps, "width": width, "height": height,
        "seconds": seconds, "speed_kmph": speed_kmph,
        "camera_height_m": config.CAMERA_HEIGHT_M,
        "camera_hfov_deg": config.CAMERA_HFOV_DEG,
        "horizon_ratio": config.HORIZON_RATIO,
        "potholes": potholes,
    }, indent=2), encoding="utf-8")
    print(f"[gen] Wrote ground truth -> {gt_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for {output_path}")

    print(f"[gen] Rendering {total_frames} frames "
          f"({seconds}s @ {fps}fps, {width}x{height}), {n_potholes} potholes...")

    for f in range(total_frames):
        travelled = speed_mps * (f / fps)
        writer.write(renderer.render(potholes, travelled, f))
        if f % (fps * 5) == 0 and f:
            print(f"      ...{f}/{total_frames} frames")

    writer.release()

    # OpenCV can only write mp4v here; shrink it to H.264 when ffmpeg exists.
    compress_h264(output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"[gen] Wrote {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic dashcam clip.")
    ap.add_argument("--seconds", type=int, default=config.SAMPLE_VIDEO_SECONDS)
    ap.add_argument("--fps", type=int, default=config.SAMPLE_VIDEO_FPS)
    ap.add_argument("--width", type=int, default=config.SAMPLE_VIDEO_WIDTH)
    ap.add_argument("--height", type=int, default=config.SAMPLE_VIDEO_HEIGHT)
    ap.add_argument("--potholes", type=int, default=config.SAMPLE_POTHOLE_COUNT)
    ap.add_argument("--out", type=str, default=str(config.INPUT_VIDEO))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    generate(Path(a.out), a.seconds, a.fps, a.width, a.height, a.potholes,
             seed=a.seed)

# ==============================================================================
# EMBEDDED MODULE: prepare_dataset.py
# ==============================================================================

# BEGIN FILE: prepare_dataset.py
########################################################################

"""
================================================================================
 prepare_dataset.py  --  Build a YOLOv8-ready pothole dataset
================================================================================

YOLOv8 expects a very specific folder layout and a `data.yaml` describing it:

    dataset/
      data.yaml
      images/train/*.jpg      labels/train/*.txt
      images/val/*.jpg        labels/val/*.txt
      images/test/*.jpg       labels/test/*.txt

Each label file holds one line per object, in NORMALISED xywh:

    <class_id> <x_center> <y_center> <width> <height>
    0 0.5124 0.7813 0.1042 0.0648

All five numbers are fractions of image width/height, NOT pixels. This is the
single most common source of silent training failure -- a model trained on pixel
coordinates converges to nonsense without ever raising an error.

WHAT THIS SCRIPT DOES
---------------------
  --split       Take a flat folder of images + labels and split it 70/20/10.
  --from-coco   Convert COCO-style JSON annotations into YOLO txt labels.
  --from-voc    Convert Pascal VOC XML annotations into YOLO txt labels.
  --synthetic   Generate a labelled synthetic dataset with zero downloads, so
                the training pipeline can be smoke-tested end to end offline.
  --verify      Sanity-check an existing dataset (orphans, malformed labels,
                out-of-range coordinates, class balance).

WHERE TO GET A REAL DATASET
---------------------------
  1. Roboflow Universe -- "pothole detection", several thousand annotated road
     images, exports directly in YOLOv8 format.
         https://universe.roboflow.com/  (search: pothole)

  2. RDD2022 (Road Damage Detection) -- ~47,000 images from India, Japan, the
     Czech Republic, Norway and the USA. Pascal VOC XML annotations, four damage
     classes including D40 (pothole).
         https://github.com/sekilab/RoadDamageDetector

  3. Kaggle -- "Pothole Detection Dataset" and "Annotated Potholes Dataset".

For RDD2022, keep only the D40 class and remap it to class 0:
    python prepare_dataset.py --from-voc RDD2022/India --keep D40 --out dataset

RUN
---
    python prepare_dataset.py --synthetic --count 800
    python prepare_dataset.py --split raw_images/ --out dataset/
    python prepare_dataset.py --verify dataset/
================================================================================
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["pothole"]


# ==============================================================================
# DATA.YAML
# ==============================================================================
def write_data_yaml(root: Path, class_names: List[str] = CLASS_NAMES) -> Path:
    """
    Write the dataset descriptor YOLO reads.

    `path` is written as an absolute path deliberately: relative paths in
    data.yaml are resolved against ultralytics' own settings directory, not the
    current working directory, which trips up almost everyone the first time.
    """
    yaml_path = root / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"# Auto-generated by prepare_dataset.py\n"
        f"path: {root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"[data] Wrote {yaml_path}")
    return yaml_path


def make_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SPLITTING
# ==============================================================================
def split_dataset(source: Path,
                  out: Path,
                  ratios: Tuple[float, float, float] = (0.70, 0.20, 0.10),
                  seed: int = 42) -> None:
    """
    Split a flat folder of images (+ matching .txt labels) into train/val/test.

    The split is done on a SHUFFLED list with a fixed seed. Shuffling matters:
    dashcam datasets are usually ordered by capture time, so an unshuffled split
    puts one continuous drive in train and a different one in val -- the val
    score then measures generalisation across roads, not across potholes, and
    swings wildly between runs.
    """
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found under {source}")

    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    buckets = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    make_dirs(out)
    missing_labels = 0

    for split, files in buckets.items():
        for img in files:
            shutil.copy2(img, out / "images" / split / img.name)

            # Labels are expected as a sibling .txt, or under a parallel
            # `labels/` folder -- both conventions are common in the wild.
            candidates = [
                img.with_suffix(".txt"),
                img.parent.parent / "labels" / (img.stem + ".txt"),
                source / "labels" / (img.stem + ".txt"),
            ]
            label = next((c for c in candidates if c.exists()), None)

            if label:
                shutil.copy2(label, out / "labels" / split / (img.stem + ".txt"))
            else:
                # A background image with no potholes is legitimate training
                # data -- YOLO uses it to learn what is NOT a pothole. Write an
                # empty label rather than skipping the image.
                (out / "labels" / split / (img.stem + ".txt")).write_text("")
                missing_labels += 1

    write_data_yaml(out)
    print(f"[split] {n} images -> train {len(buckets['train'])} | "
          f"val {len(buckets['val'])} | test {len(buckets['test'])}")
    if missing_labels:
        print(f"[split] {missing_labels} images had no label file "
              f"(written as empty = background images)")


# ==============================================================================
# FORMAT CONVERTERS
# ==============================================================================
def voc_to_yolo(voc_dir: Path,
                out: Path,
                keep_classes: List[str] | None = None) -> None:
    """
    Convert Pascal VOC XML annotations (RDD2022 format) to YOLO txt.

    VOC stores absolute pixel corners (xmin, ymin, xmax, ymax); YOLO wants
    normalised centre + size. The conversion is:

        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w   = (xmax - xmin)     / img_w
        h   = (ymax - ymin)     / img_h
    """
    import xml.etree.ElementTree as ET

    xmls = sorted(voc_dir.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No .xml annotations found under {voc_dir}")

    staging = out / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    kept, skipped = 0, 0

    for xml in xmls:
        tree = ET.parse(xml)
        root = tree.getroot()

        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)
        if img_w <= 0 or img_h <= 0:
            skipped += 1
            continue

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if keep_classes and name not in keep_classes:
                continue

            bb = obj.find("bndbox")
            xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
            xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)

            xc = (xmin + xmax) / 2.0 / img_w
            yc = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # Clamp: VOC boxes occasionally run a pixel past the image edge.
            xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
            w, h = min(max(w, 0), 1), min(max(h, 0), 1)
            if w <= 0 or h <= 0:
                continue

            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Find the matching image next to the XML.
        img = next((xml.with_suffix(e) for e in IMAGE_EXTS
                    if xml.with_suffix(e).exists()), None)
        if img is None:
            skipped += 1
            continue

        shutil.copy2(img, staging / img.name)
        (staging / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        kept += 1

    print(f"[voc] Converted {kept} annotations ({skipped} skipped)")
    split_dataset(staging, out)
    shutil.rmtree(staging, ignore_errors=True)


def coco_to_yolo(coco_json: Path, images_dir: Path, out: Path) -> None:
    """Convert COCO-style JSON annotations to YOLO txt labels."""
    blob = json.loads(coco_json.read_text(encoding="utf-8"))

    images = {im["id"]: im for im in blob["images"]}
    per_image: Dict[int, List[str]] = {}

    for ann in blob["annotations"]:
        im = images.get(ann["image_id"])
        if not im:
            continue
        # COCO bbox is [x_min, y_min, width, height] in pixels.
        x, y, w, h = ann["bbox"]
        iw, ih = im["width"], im["height"]
        per_image.setdefault(ann["image_id"], []).append(
            f"0 {(x + w / 2) / iw:.6f} {(y + h / 2) / ih:.6f} "
            f"{w / iw:.6f} {h / ih:.6f}"
        )

    staging = out / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for iid, im in images.items():
        src = images_dir / im["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, staging / src.name)
        (staging / (src.stem + ".txt")).write_text(
            "\n".join(per_image.get(iid, [])), encoding="utf-8")

    print(f"[coco] Converted {len(images)} images")
    split_dataset(staging, out)
    shutil.rmtree(staging, ignore_errors=True)


# ==============================================================================
# SYNTHETIC DATASET
# ==============================================================================
def synthetic_dataset(out: Path, count: int = 800, seed: int = 42) -> None:
    """
    Render a labelled synthetic pothole dataset with perfect ground truth.

    This exists so `train_yolo.py` can be smoke-tested with no downloads. A model
    trained purely on this will NOT generalise to real roads -- synthetic
    textures lack the shadows, wet patches, tar repairs and motion blur that make
    real detection hard. Treat it as a plumbing test, not a research result.

    That said, synthetic pretraining followed by fine-tuning on a few hundred
    real images is a legitimate and well-documented strategy when real labelled
    data is scarce.
    """
    import cv2
    import numpy as np

    from make_sample_video import RoadRenderer
    rng = random.Random(seed)
    make_dirs(out)

    W, H = 640, 384
    renderer = RoadRenderer(W, H, seed=seed)
    splits = ["train"] * 70 + ["val"] * 20 + ["test"] * 10

    for i in range(count):
        split = splits[i % 100]

        # 1-4 potholes per image, at random ranges.
        n = rng.randint(1, 4)
        potholes, boxes = [], []
        for _ in range(n):
            d = rng.uniform(3.2, 11.0)
            ph = {
                "s_m": d, "lateral_m": rng.uniform(-2.0, 2.0),
                "width_m": rng.uniform(0.35, 1.9),
                "length_m": rng.uniform(0.3, 1.7),
                "angle": rng.randint(0, 179),
                "notches": rng.randint(4, 9),
                "seedphase": rng.uniform(0, 6.28),
            }
            potholes.append(ph)

            # Derive the exact pixel box from the same projection used to draw.
            near = renderer._project_point(d, ph["lateral_m"])
            far = renderer._project_point(d + ph["length_m"], ph["lateral_m"])
            if not (near and far):
                continue
            ax = (ph["width_m"] / 2.0) * renderer.focal_px / d
            x1, x2 = near[0] - ax, near[0] + ax
            y1, y2 = far[1], near[1]
            if x2 - x1 < 6 or y2 - y1 < 3:
                continue
            boxes.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))

        frame = renderer.render(potholes, 0.0, i)

        # Photometric augmentation at generation time: brightness, contrast and
        # blur variation so the model does not latch onto one exposure.
        alpha = rng.uniform(0.72, 1.28)
        beta = rng.uniform(-24, 24)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            frame = cv2.GaussianBlur(frame, (k, k), 0)

        name = f"synth_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        lines = [
            f"0 {((b[0] + b[2]) / 2) / W:.6f} {((b[1] + b[3]) / 2) / H:.6f} "
            f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
            for b in boxes
        ]
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines), encoding="utf-8")

        if (i + 1) % 200 == 0:
            print(f"[synth] {i + 1}/{count} images")

    write_data_yaml(out)
    print(f"[synth] Wrote {count} labelled images to {out}")


# ==============================================================================
# VERIFICATION
# ==============================================================================
def verify(root: Path) -> int:
    """
    Audit a dataset before wasting GPU hours on it.

    Checks, in order of how often each one actually bites:
      1. Images with no label file (silently trained as background).
      2. Label files with no image (dead weight).
      3. Coordinates outside [0, 1] -- the pixel-vs-normalised mistake.
      4. Malformed lines (wrong field count, non-numeric).
      5. Zero-area boxes.
      6. Class balance across splits.
    """
    problems = 0
    print("=" * 64)
    print(f" VERIFYING {root}")
    print("=" * 64)

    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            print(f"  {split:<6}: MISSING")
            problems += 1
            continue

        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()

        no_label = imgs - lbls
        orphan = lbls - imgs
        boxes, bad_coords, malformed, zero_area = 0, 0, 0, 0
        classes: Counter = Counter()

        for stem in sorted(lbls & imgs):
            for ln, line in enumerate((lbl_dir / f"{stem}.txt").read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    cid = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    malformed += 1
                    continue

                boxes += 1
                classes[cid] += 1
                if any(v < 0 or v > 1 for v in vals):
                    bad_coords += 1
                if vals[2] <= 0 or vals[3] <= 0:
                    zero_area += 1

        print(f"  {split:<6}: {len(imgs):5d} images | {boxes:6d} boxes | "
              f"{boxes / max(len(imgs), 1):4.1f} boxes/img | classes {dict(classes)}")
        for label, n, note in [
            ("images without labels", len(no_label), "treated as background"),
            ("orphan label files", len(orphan), "no matching image"),
            ("coords outside [0,1]", bad_coords, "PIXEL COORDS? must be normalised"),
            ("malformed lines", malformed, "expected 5 fields"),
            ("zero-area boxes", zero_area, "will be dropped by YOLO"),
        ]:
            if n:
                print(f"           ! {n} {label}  ({note})")
                problems += n

    print("=" * 64)
    print("  Dataset looks healthy." if problems == 0
          else f"  {problems} issue(s) found -- fix before training.")
    return 0 if problems == 0 else 1


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare a YOLOv8 pothole dataset.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", type=str, help="Flat folder of images to split")
    g.add_argument("--from-voc", type=str, help="Pascal VOC directory (RDD2022)")
    g.add_argument("--from-coco", type=str, help="COCO annotations JSON")
    g.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    g.add_argument("--verify", type=str, help="Audit an existing dataset")

    ap.add_argument("--out", type=str, default="dataset", help="Output dataset root")
    ap.add_argument("--images", type=str, default=None, help="Image dir (for COCO)")
    ap.add_argument("--keep", nargs="*", default=None,
                    help="VOC class names to keep, e.g. --keep D40")
    ap.add_argument("--count", type=int, default=800, help="Synthetic image count")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)

    if a.verify:
        return verify(Path(a.verify))
    if a.synthetic:
        synthetic_dataset(out, a.count, a.seed)
    elif a.split:
        split_dataset(Path(a.split), out, seed=a.seed)
    elif a.from_voc:
        voc_to_yolo(Path(a.from_voc), out, a.keep)
    elif a.from_coco:
        if not a.images:
            raise SystemExit("--from-coco also requires --images <dir>")
        coco_to_yolo(Path(a.from_coco), Path(a.images), out)

    return verify(out)


if __name__ == "__main__":
    raise SystemExit(main())


########################################################################
# END FILE: prepare_dataset.py
########################################################################

########################################################################

# ==============================================================================
# EMBEDDED MODULE: train_yolo.py
# ==============================================================================

# BEGIN FILE: train_yolo.py
########################################################################

"""
================================================================================
 train_yolo.py  --  Train a pothole detector with YOLOv8
================================================================================

QUICK START
-----------
    pip install ultralytics
    python prepare_dataset.py --synthetic --count 1000     # or a real dataset
    python train_yolo.py --data dataset/data.yaml --epochs 100

    # then just run the pipeline -- it picks the weights up automatically
    python main.py

The trained weights are copied to `models/pothole_yolov8.pt`, which is exactly
where `config.YOLO_WEIGHTS` points, so `main.py` switches from the simulated
detector to real inference with no further configuration.

MODEL SIZE -- pick for your deployment target
---------------------------------------------
    yolov8n   3.2M params   fastest   Raspberry Pi / Jetson Nano dashcam
    yolov8s  11.2M params   balanced  RECOMMENDED starting point
    yolov8m  25.9M params   accurate  desktop GPU, offline survey processing
    yolov8l  43.7M params   slow      research baselines only

TRANSFER LEARNING
-----------------
We always start from a COCO-pretrained checkpoint rather than random weights.
COCO has no pothole class, but its early layers have already learned edges,
textures and shadow gradients from 330k images. Fine-tuning those beats training
from scratch by a wide margin on datasets under ~10k images -- which is every
publicly available pothole dataset.

AUGMENTATION -- tuned for road imagery specifically
---------------------------------------------------
The defaults are tuned for COCO (people, cars, animals) and several actively
hurt here. What we change and why:

  fliplr=0.5      KEEP. A pothole is mirror-symmetric in meaning; horizontal
                  flips double the data for free.
  flipud=0.0      OFF. An upside-down road never occurs. Vertical flips teach
                  the model that potholes can appear in the sky.
  degrees=5       SMALL. Real dashcams roll a few degrees on cornering and
                  uneven camber, but never 45 degrees.
  perspective     SMALL but non-zero. This is the one augmentation that matters
                  most here: it simulates different camera mounting heights and
                  pitch angles, which is precisely the variation between cars.
  hsv_v=0.5       HIGH. Lighting is the dominant nuisance variable on roads --
                  bright noon glare, overcast, tunnel shadow, dusk.
  hsv_s=0.6       HIGH. Wet asphalt, dry asphalt and fresh tar differ hugely in
                  saturation.
  mosaic=1.0      ON. Stitches 4 images together; strongly improves small-object
                  detection, and distant potholes ARE small objects.
  close_mosaic=10 Disable mosaic for the final 10 epochs so the model finishes
                  training on undistorted, realistic full frames.
  scale=0.5       WIDE. Potholes span a huge apparent size range (25 m away vs
                  3 m away), so aggressive scale jitter is essential.

RUN
---
    python train_yolo.py --data dataset/data.yaml --model yolov8s.pt --epochs 100
    python train_yolo.py --data dataset/data.yaml --resume
    python train_yolo.py --validate-only --weights models/pothole_yolov8.pt
    python train_yolo.py --export onnx --weights models/pothole_yolov8.pt
================================================================================
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ==============================================================================
def require_ultralytics():
    """Import ultralytics with an actionable error instead of a stack trace."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("=" * 68)
        print(" `ultralytics` is not installed.")
        print("=" * 68)
        print(" Install it with:\n")
        print("     pip install ultralytics\n")
        print(" On a CUDA machine, install PyTorch first for GPU support:")
        print("     https://pytorch.org/get-started/locally/\n")
        sys.exit(1)


def detect_device() -> str:
    """Return the best available device string for ultralytics."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA available: {name}")
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[device] Apple MPS available")
            return "mps"
    except ImportError:
        pass
    print("[device] No GPU found -- training on CPU. Expect this to be slow;")
    print("         consider Google Colab or Kaggle for a free T4/P100.")
    return "cpu"


# ==============================================================================
# AUGMENTATION PROFILE
# ==============================================================================
ROAD_AUGMENTATION = dict(
    # --- geometric -------------------------------------------------------
    fliplr=0.5,        # horizontal flip: safe and free data
    flipud=0.0,        # vertical flip: physically impossible, would hurt
    degrees=5.0,       # small roll, matching real camber and cornering
    translate=0.12,    # camera not perfectly centred in the lane
    scale=0.5,         # wide: potholes vary enormously in apparent size
    shear=2.0,
    perspective=0.0004,  # simulates different mounting heights / pitch

    # --- photometric -----------------------------------------------------
    hsv_h=0.015,       # hue barely varies on asphalt; keep this low
    hsv_s=0.6,         # wet vs dry vs freshly tarred road
    hsv_v=0.5,         # glare, overcast, tunnel shadow, dusk

    # --- composition -----------------------------------------------------
    mosaic=1.0,        # 4-image stitching: big win for small objects
    close_mosaic=10,   # last 10 epochs on clean frames
    mixup=0.1,         # light: heavy mixup blurs the fine rim texture
    erasing=0.2,       # random occlusion -> robustness to puddles and cars
)


# ==============================================================================
def train(data_yaml: Path,
          base_model: str = config.YOLO_BASE_CHECKPOINT,
          epochs: int = 100,
          imgsz: int = 640,
          batch: int = 16,
          patience: int = 25,
          workers: int = 4,
          project: str = "runs",
          name: str = "pothole_v8",
          resume: bool = False,
          device: str | None = None):
    """
    Fine-tune YOLOv8 on the pothole dataset and install the best weights.

    Key hyper-parameter notes:

    batch=16    On an 8 GB GPU at imgsz 640 this fits comfortably. Pass
                batch=-1 to let ultralytics auto-size to ~60% VRAM.
    patience=25 Early stopping. Pothole datasets are small and overfit fast;
                without patience the model memorises the training roads.
    imgsz=640   Do not drop below this. Distant potholes are already only a few
                dozen pixels tall; downscaling erases them entirely.
    cos_lr      Cosine learning-rate decay, which consistently outperforms the
                default linear schedule on small fine-tuning runs.
    """
    YOLO = require_ultralytics()
    device = device or detect_device()

    if not Path(data_yaml).exists():
        raise SystemExit(
            f"data.yaml not found: {data_yaml}\n"
            f"Create a dataset first:\n"
            f"    python prepare_dataset.py --synthetic --count 1000"
        )

    print("=" * 68)
    print(" TRAINING YOLOv8 POTHOLE DETECTOR")
    print("=" * 68)
    print(f"  Base checkpoint : {base_model}")
    print(f"  Dataset         : {data_yaml}")
    print(f"  Epochs          : {epochs}   (early stop patience {patience})")
    print(f"  Image size      : {imgsz}")
    print(f"  Batch size      : {batch}")
    print(f"  Device          : {device}")
    print("=" * 68)

    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
        project=project,
        name=name,
        resume=resume,
        exist_ok=True,

        # --- optimisation -------------------------------------------------
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,       # cosine decay beats the linear default here
        warmup_epochs=3.0,
        weight_decay=0.0005,

        # --- loss weighting ------------------------------------------------
        # box loss is raised above the 7.5 default: for our downstream depth
        # and area estimation, box TIGHTNESS matters more than classification
        # confidence. A loose box directly becomes a wrong depth reading.
        box=9.0,
        cls=0.5,
        dfl=1.5,

        # --- bookkeeping ---------------------------------------------------
        val=True,
        plots=True,
        save=True,
        save_period=25,
        seed=42,
        deterministic=True,
        **ROAD_AUGMENTATION,
    )

    # ---- install the best weights where the pipeline expects them ---------
    best = Path(project) / name / "weights" / "best.pt"
    if best.exists():
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, config.YOLO_WEIGHTS)
        print(f"\n[train] Installed best weights -> {config.YOLO_WEIGHTS}")
        print("[train] `python main.py` will now use real YOLO inference.")
    else:
        print(f"\n[train] WARNING: expected weights at {best} but none were found.")

    return results


# ==============================================================================
def validate(weights: Path, data_yaml: Path, imgsz: int = 640, split: str = "test"):
    """
    Report detection metrics on a held-out split.

    HOW TO READ THE NUMBERS
    -----------------------
    mAP50      Mean average precision at IoU 0.50. The headline number most
               papers quote. Above ~0.70 is a usable pothole detector.
    mAP50-95   Averaged over IoU 0.50:0.95. Much stricter -- it rewards TIGHT
               boxes. This is the metric that actually matters for us, because
               depth and area are computed from the box dimensions, so a loose
               box that still counts as a hit at IoU 0.5 produces a badly wrong
               severity score.
    Precision  Of the things called potholes, how many were. Low precision means
               the map fills with phantom markers and crews lose trust in it.
    Recall     Of the real potholes, how many were found. Low recall means
               dangerous holes are silently missed.
    """
    YOLO = require_ultralytics()

    if not Path(weights).exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split=split, plots=True)

    print("=" * 68)
    print(f" VALIDATION METRICS  ({split} split)")
    print("=" * 68)
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}   <- box tightness, matters most here")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("=" * 68)
    return metrics


# ==============================================================================
def export(weights: Path, fmt: str = "onnx", imgsz: int = 640):
    """
    Export to a deployment format.

    onnx        Cross-platform, runs under ONNXRuntime on CPU. Best default for
                an in-vehicle box without a GPU.
    engine      NVIDIA TensorRT. Large speedup on Jetson hardware; must be built
                on the exact device it will run on.
    tflite      Android / edge microcontrollers.
    coreml      iOS dashcam apps.
    openvino    Intel CPUs and NPUs; strong CPU speedup on x86.
    """
    YOLO = require_ultralytics()
    model = YOLO(str(weights))
    path = model.export(format=fmt, imgsz=imgsz, simplify=True)
    print(f"[export] Wrote {path}")
    return path


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train / validate / export a YOLOv8 pothole detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--model", type=str, default=config.YOLO_BASE_CHECKPOINT,
                    help="Base checkpoint: yolov8n/s/m/l.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16,
                    help="Use -1 for automatic batch sizing")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cpu, 0, 0,1, mps")
    ap.add_argument("--name", type=str, default="pothole_v8")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--weights", type=str, default=str(config.YOLO_WEIGHTS))
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--export", type=str, default=None,
                    choices=["onnx", "engine", "tflite", "coreml", "openvino", "torchscript"])
    a = ap.parse_args()

    if a.export:
        export(Path(a.weights), a.export, a.imgsz)
        return 0

    if a.validate_only:
        validate(Path(a.weights), Path(a.data), a.imgsz, a.split)
        return 0

    train(Path(a.data), a.model, a.epochs, a.imgsz, a.batch,
          a.patience, a.workers, name=a.name, resume=a.resume, device=a.device)

    # Automatically score the freshly trained model on the held-out test split.
    if config.YOLO_WEIGHTS.exists():
        print("\n[train] Scoring the trained model on the test split...")
        validate(config.YOLO_WEIGHTS, Path(a.data), a.imgsz, "test")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


########################################################################
# END FILE: train_yolo.py
########################################################################

# ==============================================================================
# EMBEDDED MODULE: evaluate.py
# ==============================================================================

"""
================================================================================
 evaluate.py  --  Quantitative accuracy report against ground truth
================================================================================

A demo that only prints "13 potholes found" proves nothing. This script answers
the question a reviewer will actually ask: *how many of them were real, and how
close were the measurements?*

It compares `outputs/pothole_clusters.csv` against the ground-truth sidecar
`data/input_dashcam.groundtruth.json` written by make_sample_video.py, and
reports:

  DETECTION QUALITY   precision, recall, F1 -- matching clusters to true
                      potholes by nearest-neighbour within a tolerance radius.

  CLUSTERING QUALITY  how well DBSCAN collapsed the raw per-frame log. The
                      ideal is one cluster per true pothole; over-splitting and
                      over-merging are both reported.

  MEASUREMENT ERROR   MAE and bias of the recovered surface area against the
                      true seeded area. This is the honest test of the
                      projection stage: we know exactly how big each pothole is,
                      so we can check what the geometry recovered from pixels.

RUN
---
    python main.py            # produce the outputs first
    python evaluate.py
================================================================================
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

MATCH_TOLERANCE_M = 6.0     # a cluster within this range counts as a true match


# ==============================================================================
def true_pothole_positions(gt: dict) -> pd.DataFrame:
    """
    Reconstruct the absolute coordinate of every seeded pothole.

    The ground truth stores each pothole as (distance along road, lateral
    offset). We replay the same GPS track the pipeline used and place each
    pothole at the point where the vehicle would have been level with it.
    """
    track = GPSTrack.from_config()
    rows = []

    for i, p in enumerate(gt["potholes"]):
        # Time at which the car draws level with this pothole.
        t = p["s_m"] / track.speed_mps
        fix = track.fix_at(t, jitter=False)      # noiseless: this is truth
        lat, lon = project_pothole(fix, 0.0, p["lateral_m"])

        rows.append({
            "true_id": i + 1,
            "latitude": lat,
            "longitude": lon,
            "s_m": p["s_m"],
            "width_m": p["width_m"],
            "length_m": p["length_m"],
            "area_m2": p["width_m"] * p["length_m"],
            "in_view_window": p["s_m"] <= track.speed_mps * gt["seconds"] + 24.0,
        })
    return pd.DataFrame(rows)


# ==============================================================================
def match_clusters(clusters: pd.DataFrame,
                   truth: pd.DataFrame,
                   tol_m: float = MATCH_TOLERANCE_M) -> pd.DataFrame:
    """
    Greedy nearest-neighbour matching between detected clusters and true holes.

    Greedy is appropriate here rather than the Hungarian algorithm: the true
    potholes are separated by more than twice the tolerance, so the assignment
    is unambiguous and greedy is provably optimal in that regime.
    """
    unmatched = set(truth["true_id"])
    records = []

    # Sort by hit_count so the most confident clusters claim their match first.
    for _, c in clusters.sort_values("hit_count", ascending=False).iterrows():
        best_id, best_d = None, float("inf")
        for _, t in truth[truth["true_id"].isin(unmatched)].iterrows():
            d = haversine_m((c["latitude"], c["longitude"]),
                            (t["latitude"], t["longitude"]))
            if d < best_d:
                best_id, best_d = t["true_id"], d

        matched = best_id is not None and best_d <= tol_m
        if matched:
            unmatched.discard(best_id)

        records.append({
            "cluster_id": c["cluster_id"],
            "matched_true_id": best_id if matched else None,
            "position_error_m": round(best_d, 2) if matched else None,
            "detected_area_m2": c["area_m2"],
            "detected_depth_cm": c["depth_cm"],
            "severity": c["severity"],
            "hit_count": c["hit_count"],
            "is_true_positive": matched,
        })

    return pd.DataFrame(records)


# ==============================================================================
def main() -> int:
    gt_path = config.INPUT_VIDEO.with_suffix(".groundtruth.json")
    if not gt_path.exists():
        print(f"No ground truth at {gt_path}.")
        print("Ground truth only exists for generated sample videos. Run:")
        print("    python make_sample_video.py && python main.py")
        return 1

    if not Path(config.CLUSTER_CSV).exists():
        print(f"No cluster output at {config.CLUSTER_CSV}. Run `python main.py` first.")
        return 1

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    clusters = pd.read_csv(config.CLUSTER_CSV)
    raw = pd.read_csv(config.LOG_CSV)
    truth = true_pothole_positions(gt)

    # Only potholes the camera actually had a chance to see count for recall.
    visible_truth = truth[truth["in_view_window"]].copy()

    matches = match_clusters(clusters, visible_truth)

    tp = int(matches["is_true_positive"].sum())
    fp = int(len(matches) - tp)
    fn = int(len(visible_truth) - tp)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print("=" * 68)
    print(" EVALUATION AGAINST GROUND TRUTH")
    print("=" * 68)
    print(f"  True potholes in camera view : {len(visible_truth)}")
    print(f"  Clusters reported            : {len(clusters)}")
    print()
    print("  DETECTION QUALITY")
    print(f"    True positives   : {tp}")
    print(f"    False positives  : {fp}")
    print(f"    False negatives  : {fn}")
    print(f"    Precision        : {precision:6.1%}")
    print(f"    Recall           : {recall:6.1%}")
    print(f"    F1 score         : {f1:6.3f}")

    # ---- localisation accuracy -------------------------------------------
    errs = matches.loc[matches["is_true_positive"], "position_error_m"].dropna()
    if len(errs):
        print()
        print("  LOCALISATION ERROR (metres from true position)")
        print(f"    Mean             : {errs.mean():6.2f} m")
        print(f"    Median           : {errs.median():6.2f} m")
        print(f"    95th percentile  : {errs.quantile(0.95):6.2f} m")

    # ---- clustering efficiency -------------------------------------------
    print()
    print("  CLUSTERING EFFICIENCY")
    print(f"    Raw detections   : {len(raw)}")
    print(f"    After DBSCAN     : {len(clusters)}")
    print(f"    Compression      : {100 * (1 - len(clusters) / max(len(raw), 1)):5.1f}%")
    print(f"    Detections/hole  : {len(raw) / max(len(clusters), 1):5.1f} avg")

    # ---- measurement accuracy --------------------------------------------
    ok = matches[matches["is_true_positive"]].copy()
    if len(ok):
        joined = ok.merge(
            visible_truth[["true_id", "area_m2"]].rename(
                columns={"area_m2": "true_area_m2"}),
            left_on="matched_true_id", right_on="true_id", how="left")
        joined = joined.dropna(subset=["true_area_m2"])

        if len(joined):
            err = joined["detected_area_m2"] - joined["true_area_m2"]
            rel = err / joined["true_area_m2"]
            print()
            print("  AREA MEASUREMENT ERROR (pixels -> m^2)")
            print(f"    Mean abs error   : {err.abs().mean():6.3f} m2")
            print(f"    Mean rel error   : {rel.abs().mean():6.1%}")
            print(f"    Bias             : {err.mean():+6.3f} m2 "
                  f"({'over' if err.mean() > 0 else 'under'}-estimating)")

    # ---- severity distribution -------------------------------------------
    print()
    print("  SEVERITY DISTRIBUTION "
          f"(reference vehicle: {clusters['vehicle'].iloc[0]})")
    for sev in ("High", "Medium", "Low"):
        n = int((clusters["severity"] == sev).sum())
        bar = "#" * n
        print(f"    {sev:<7}: {n:3d}  {bar}")
    print("=" * 68)

    out = config.OUTPUT_DIR / "evaluation_matches.csv"
    matches.to_csv(out, index=False)
    print(f"  Per-cluster detail written to {out.name}")
    return 0

# ==============================================================================
# EMBEDDED MODULE: main.py
# ==============================================================================

# BEGIN FILE: main.py
########################################################################

"""
================================================================================
 main.py  --  End-to-end Pothole Detection & GPS Impact Analysis pipeline
================================================================================

PIPELINE
--------
    input_dashcam.mp4
          |
          v
    [1] DETECT      YOLOv8 (or SimulatedDetector) -> pixel bounding boxes
          |
          v
    [2] MEASURE     ground-plane projection -> area, distance, depth proxy
          |
          v
    [3] SCORE       depth + area + speed + vehicle -> ISI -> Low/Medium/High
          |
          v
    [4] GEOTAG      simulated GNSS fix at this timestamp -> lat/lon
          |
          +--> [5a] RENDER    annotated frame -> output_annotated.mp4
          +--> [5b] ALERT     beep + banner + alerts.log if severity == High
          +--> [5c] LOG       one row -> pothole_log.csv
                                   |
                                   v
                           [6] CLUSTER   DBSCAN @ 3 m -> unique potholes
                                   |
                                   v
                           [7] MAP       folium -> pothole_map.html

USAGE
-----
    python main.py                                # full run, defaults
    python main.py --vehicle SUV                  # re-score for an SUV
    python main.py --input my_drive.mp4           # your own footage
    python main.py --radius 5                     # wider merge radius
    python main.py --no-video                     # skip rendering (fast)
    python main.py --preview                      # live OpenCV window
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

# Make `src` importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ==============================================================================
# DEPENDENCY GUARD -- fail with instructions, never with a raw traceback
# ==============================================================================
def check_dependencies() -> dict:
    """
    Verify what is installed and report clearly.

    Hard requirements  : numpy, opencv-python, pandas
    Soft requirements  : scikit-learn (clustering), folium (map),
                         ultralytics (real detection), simpleaudio (beeps)
    """
    status, missing_hard = {}, []

    for mod, pkg, hard in [
        ("numpy", "numpy", True),
        ("cv2", "opencv-python", True),
        ("pandas", "pandas", True),
        ("sklearn", "scikit-learn", False),
        ("folium", "folium", False),
        ("ultralytics", "ultralytics", False),
        ("simpleaudio", "simpleaudio", False),
    ]:
        try:
            __import__(mod)
            status[pkg] = True
        except ImportError:
            status[pkg] = False
            if hard:
                missing_hard.append(pkg)

    if missing_hard:
        print("\n" + "=" * 68)
        print(" MISSING REQUIRED PACKAGES")
        print("=" * 68)
        print(" Install them with:\n")
        print(f"     pip install {' '.join(missing_hard)}\n")
        sys.exit(1)

    optional_missing = [p for p, ok in status.items()
                        if not ok and p not in ("numpy", "opencv-python", "pandas")]
    if optional_missing:
        print(f"[deps] Optional packages not found: {', '.join(optional_missing)}")
        print(f"[deps] The pipeline will degrade gracefully. To enable everything:")
        print(f"       pip install {' '.join(optional_missing)}")
    return status


# ==============================================================================
def process_video(input_path: Path,
                  output_path: Path,
                  write_video: bool = True,
                  preview: bool = False,
                  max_frames: int | None = None):
    """
    Stages 1-5: read the video, detect, measure, score, geotag, render, log.

    Returns
    -------
    pandas.DataFrame
        The raw per-frame detection log (also written to pothole_log.csv).
    """
    import cv2
    import pandas as pd

    from src import renderer

    # ---- open the source video -------------------------------------------
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}\n"
            f"Generate a sample with:  python make_sample_video.py"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or config.SAMPLE_VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = n_frames / fps if fps else 0.0

    print(f"[video] {input_path.name}: {width}x{height} @ {fps:.1f}fps, "
          f"{n_frames} frames ({duration_s:.1f}s)")

    # ---- build the subsystems --------------------------------------------
    vehicle = config.get_vehicle()
    track = GPSTrack.from_config()
    projector = GroundPlaneProjector(width, height)
    alerts = AlertManager()
    detector = load_detector(width, height, track.speed_mps, duration_s or 20.0)

    print(f"[setup] {track.describe()}")
    print(f"[setup] Vehicle: {vehicle.name} "
          f"(clearance {vehicle.ground_clearance_cm}cm, "
          f"bridging {vehicle.bridging_capacity_cm:.1f}cm)")

    # ---- output writer ----------------------------------------------------
    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[video] WARNING: cannot write {output_path}; continuing without video.")
            writer = None

    # ---- main loop --------------------------------------------------------
    rows = []
    counts = {"Low": 0, "Medium": 0, "High": 0}
    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        timestamp_s = frame_idx / fps
        fix = track.fix_at(timestamp_s)              # [4] GEOTAG

        renderer.draw_horizon(frame)

        detections = detector.detect(frame, frame_idx, timestamp_s)   # [1] DETECT
        frame_has_severe = False

        for det in detections:
            # [2] MEASURE + [3] SCORE
            a = assess(det.box, projector, fix.speed_kmph, vehicle)
            if a is None:                # too far away / above the horizon
                continue

            # Geolocate the POTHOLE, not the car. See gps_simulator.
            ph_lat, ph_lon = project_pothole(fix, a.distance_m, a.lateral_m)

            counts[a.severity] += 1

            # [5a] RENDER
            if writer is not None or preview:
                renderer.draw_detection(frame, det.box, det.confidence, a, det.track_id)

            if a.severity == "High":
                frame_has_severe = True

            # [5c] LOG -- one row per detection per frame
            rows.append({
                "frame_index": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                # The pothole's own coordinate -- this is what gets clustered.
                "latitude": ph_lat,
                "longitude": ph_lon,
                # The car's coordinate, kept for auditing / route reconstruction.
                "vehicle_lat": fix.latitude,
                "vehicle_lon": fix.longitude,
                "speed_kmph": fix.speed_kmph,
                "heading_deg": fix.heading_deg,
                "track_id": det.track_id,
                "confidence": round(det.confidence, 3),
                "x1": round(det.x1, 1), "y1": round(det.y1, 1),
                "x2": round(det.x2, 1), "y2": round(det.y2, 1),
                "box_h_ratio": round(det.height / height, 4),
                "distance_m": a.distance_m,
                "lateral_m": a.lateral_m,
                "width_m": a.width_m,
                "length_m": a.length_m,
                "area_m2": a.area_m2,
                "depth_cm": a.depth_cm,
                "isi": a.isi,
                "severity": a.severity,
                "vehicle": a.vehicle,
                "detector": detector.name,
            })

        # [5b] ALERT
        if frame_has_severe:
            worst = max((r for r in rows[-len(detections):] if r["severity"] == "High"),
                        key=lambda r: r["isi"], default=None)
            fired = alerts.maybe_alert("High", timestamp_s, worst or {})
            if (writer is not None or preview) and (fired or frame_has_severe):
                renderer.draw_warning(frame, frame_idx)

        if writer is not None or preview:
            renderer.draw_hud(frame, fix, counts, frame_idx, fps, detector.name)

        if writer is not None:
            writer.write(frame)

        if preview:
            cv2.imshow("Pothole-AI (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if n_frames and frame_idx % max(1, n_frames // 10) == 0:
            pct = 100 * frame_idx / n_frames
            print(f"[video] {pct:5.1f}%  ({frame_idx}/{n_frames})  "
                  f"detections so far: {len(rows)}")

    # ---- teardown ---------------------------------------------------------
    cap.release()
    if writer is not None:
        writer.release()
        compress_h264(output_path)
        print(f"[video] Wrote annotated video -> {output_path}")
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    proc_fps = frame_idx / elapsed if elapsed else 0
    print(f"[video] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({proc_fps:.1f} fps, {proc_fps / max(fps, 1):.1f}x real time)")
    print(f"[alert] {alerts.summary()}")

    # ---- write the raw log ------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(config.LOG_CSV, index=False)
    print(f"[log] {len(df)} raw detections -> {config.LOG_CSV}")
    if not df.empty:
        dist = df["severity"].value_counts().to_dict()
        print(f"[log] Severity mix: {dist}")
    return df


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real-Time Pothole Detection & GPS Impact Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", type=str, default=str(config.INPUT_VIDEO),
                    help="Source dashcam video")
    ap.add_argument("--output", type=str, default=str(config.OUTPUT_VIDEO),
                    help="Annotated output video")
    ap.add_argument("--vehicle", type=str, default=config.VEHICLE_TYPE,
                    choices=list(config.VEHICLE_PROFILES),
                    help="Reference vehicle for the Impact Severity Index")
    ap.add_argument("--radius", type=float, default=config.CLUSTER_RADIUS_M,
                    help="DBSCAN merge radius in metres")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip video rendering (much faster)")
    ap.add_argument("--no-map", action="store_true", help="Skip HTML map")
    ap.add_argument("--preview", action="store_true", help="Live OpenCV window")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process only the first N frames (debugging)")
    ap.add_argument("--simulate", action="store_true",
                    help="Force the simulated detector even if weights exist")
    args = ap.parse_args()

    print("=" * 68)
    print(" REAL-TIME POTHOLE DETECTION & GPS IMPACT ANALYSIS")
    print("=" * 68)

    check_dependencies()

    # Apply CLI overrides to the global config before anything imports it.
    config.VEHICLE_TYPE = args.vehicle
    if args.simulate:
        config.FORCE_SIMULATION = True

    # ---- ensure we have input footage ------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[input] {input_path} not found -- generating a synthetic clip.")
        from make_sample_video import generate
        input_path = generate(input_path)

    # ---- stages 1-5 --------------------------------------------------------
    df = process_video(
        input_path=input_path,
        output_path=Path(args.output),
        write_video=not args.no_video,
        preview=args.preview,
        max_frames=args.max_frames,
    )

    if df.empty:
        print("\n[done] No potholes detected -- nothing to cluster or map.")
        return 0

    # ---- stage 6: cluster --------------------------------------------------
    print("-" * 68)
    clusters = cluster_csv(radius_m=args.radius)

    # ---- stage 7: map ------------------------------------------------------
    if clusters is not None and not args.no_map:
        print("-" * 68)
        build_map(clusters, route=config.ROUTE_WAYPOINTS)

    # ---- summary -----------------------------------------------------------
    print("=" * 68)
    print(" RUN SUMMARY")
    print("=" * 68)
    if clusters is not None and not clusters.empty:
        by_sev = clusters["severity"].value_counts().to_dict()
        verified = int(clusters["verified"].sum())
        print(f"  Reference vehicle    : {config.VEHICLE_TYPE}")
        print(f"  Raw detections       : {len(df)}")
        print(f"  Unique potholes      : {len(clusters)}  ({verified} verified)")
        print(f"  Severity breakdown   : High={by_sev.get('High', 0)}  "
              f"Medium={by_sev.get('Medium', 0)}  Low={by_sev.get('Low', 0)}")
        print(f"  Deepest pothole      : {clusters['depth_cm'].max():.1f} cm")
        print(f"  Mean depth           : {clusters['depth_cm'].mean():.1f} cm")
    print("\n  Artefacts written to outputs/:")
    for p in (config.OUTPUT_VIDEO, config.LOG_CSV, config.CLUSTER_CSV,
              config.MAP_HTML, config.ALERT_LOG):
        if Path(p).exists():
            print(f"    - {Path(p).name:<26} {Path(p).stat().st_size / 1024:8.1f} KB")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


########################################################################
# END FILE: main.py
########################################################################

########################################################################
