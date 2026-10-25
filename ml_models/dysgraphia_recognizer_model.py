"""
Dysgraphia Drawing Recognition ML Model
Uses Scikit-Learn Neural Networks (MLPClassifier) and Feature Extraction
to recognize handwritten geometric shapes (Triangle, Square, Circle, Line, Rectangle)
and alphabet characters ('A', 'B', 'C', 'D', 'P', 'R', etc.)
Lightweight, ultra-fast (<5ms), fits perfectly within Vercel's limits.
"""

import os
import math
import joblib
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_MODEL_PATH = os.path.join(MODEL_DIR, "dysgraphia_shape_model.joblib")
LETTER_MODEL_PATH = os.path.join(MODEL_DIR, "dysgraphia_letter_model.joblib")

ONNX_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "alphabet_model.onnx"),
    os.path.join(os.path.dirname(MODEL_DIR), "alphabet-classifier", "outputs", "exports", "alphabet_model.onnx")
]
ONNX_MODEL_PATH = next((p for p in ONNX_MODEL_CANDIDATES if os.path.isfile(p)), ONNX_MODEL_CANDIDATES[0])

SHAPE_CLASSES = ["triangle", "square", "circle", "line", "rectangle"]
LETTER_CLASSES = ["A", "B", "C", "D", "P", "R", "S", "T", "O", "L"]
ONNX_ALPHA_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["blank"]

_onnx_letter_session = None


def get_onnx_letter_session():
    """Load and cache the ONNX MobileNetV2 alphabet classifier."""
    global _onnx_letter_session
    if _onnx_letter_session is None and ONNX_AVAILABLE:
        if os.path.exists(ONNX_MODEL_PATH):
            try:
                print(f"[*] Initializing ONNX MobileNetV2 Alphabet Classifier from: {ONNX_MODEL_PATH}")
                _onnx_letter_session = ort.InferenceSession(ONNX_MODEL_PATH)
            except Exception as e:
                print(f"[!] Could not load ONNX session: {e}")
    return _onnx_letter_session


def extract_points_from_stroke(stroke: Any) -> List[Tuple[float, float]]:
    """Helper to extract (x, y) tuples from any stroke format (dict, list, nested)."""
    pts = []
    if not stroke:
        return pts
    
    # 1. Dict format
    if isinstance(stroke, dict):
        if "points" in stroke and isinstance(stroke["points"], (list, tuple)):
            for pt in stroke["points"]:
                if isinstance(pt, dict):
                    pts.append((float(pt.get("x", 0)), float(pt.get("y", 0))))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append((float(pt[0]), float(pt[1])))
            return pts
        elif "coordinates" in stroke or "path" in stroke:
            raw = stroke.get("coordinates") or stroke.get("path") or []
            for pt in raw:
                if isinstance(pt, dict):
                    pts.append((float(pt.get("x", 0)), float(pt.get("y", 0))))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append((float(pt[0]), float(pt[1])))
            return pts
        elif "x" in stroke or "y" in stroke:
            return [(float(stroke.get("x", 0)), float(stroke.get("y", 0)))]

    # 2. List or tuple format
    if isinstance(stroke, (list, tuple)):
        for item in stroke:
            if isinstance(item, dict):
                if "points" in item or "coordinates" in item or "path" in item:
                    pts.extend(extract_points_from_stroke(item))
                elif "x" in item or "y" in item:
                    pts.append((float(item.get("x", 0)), float(item.get("y", 0))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if all(isinstance(v, (int, float)) for v in item[:2]):
                    pts.append((float(item[0]), float(item[1])))
                else:
                    pts.extend(extract_points_from_stroke(item))
    return pts


def normalize_strokes(strokes: List[Any]) -> List[List[Tuple[float, float]]]:
    """
    Extract (x, y) coordinates from arbitrary stroke format (tuples, dicts, arrays)
    and normalize them into a centered [0, 1] x [0, 1] bounding box.
    """
    cleaned_strokes = []
    all_x = []
    all_y = []

    for stroke in strokes:
        curr = extract_points_from_stroke(stroke)
        if curr:
            cleaned_strokes.append(curr)
            for x, y in curr:
                all_x.append(x)
                all_y.append(y)

    if not all_x or not all_y:
        return []

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max(max_x - min_x, 1e-4)
    height = max(max_y - min_y, 1e-4)
    max_dim = max(width, height)

    normalized = []
    for stroke in cleaned_strokes:
        norm_stroke = []
        for x, y in stroke:
            # Aspect-ratio preserving normalization into [0.1, 0.9]
            nx = 0.1 + 0.8 * ((x - min_x) / max_dim)
            ny = 0.1 + 0.8 * ((y - min_y) / max_dim)
            norm_stroke.append((nx, ny))
        normalized.append(norm_stroke)

    return normalized


def rasterize_strokes_to_grid(strokes: List[List[Tuple[float, float]]], grid_size: int = 16) -> np.ndarray:
    """
    Render strokes onto a grid_size x grid_size grayscale density grid (16x16 = 256 features).
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if not strokes:
        return grid.flatten()

    for stroke in strokes:
        if len(stroke) == 1:
            gx = int(np.clip(stroke[0][0] * grid_size, 0, grid_size - 1))
            gy = int(np.clip(stroke[0][1] * grid_size, 0, grid_size - 1))
            grid[gy, gx] = 1.0
            continue

        for i in range(1, len(stroke)):
            p0 = stroke[i - 1]
            p1 = stroke[i]
            # Interpolate points along the line segment
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            steps = max(1, int(dist * grid_size * 2.5))
            for s in range(steps + 1):
                t = s / steps
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                gx = int(np.clip(x * grid_size, 0, grid_size - 1))
                gy = int(np.clip(y * grid_size, 0, grid_size - 1))
                grid[gy, gx] = 1.0
                # Slight anti-aliasing / brush dilation
                if gx + 1 < grid_size:
                    grid[gy, gx + 1] = max(grid[gy, gx + 1], 0.4)
                if gy + 1 < grid_size:
                    grid[gy + 1, gx] = max(grid[gy + 1, gx], 0.4)

    return grid.flatten()


def extract_geometric_features(strokes: List[List[Tuple[float, float]]]) -> np.ndarray:
    """
    Extract 16 geometric, invariant, and kinematic features from normalized strokes.
    """
    if not strokes:
        return np.zeros(16, dtype=np.float32)

    all_pts = [pt for stroke in strokes for pt in stroke]
    if len(all_pts) < 2:
        return np.zeros(16, dtype=np.float32)

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-4)
    height = max(max_y - min_y, 1e-4)

    # 1. Aspect ratio
    aspect_ratio = width / height

    # 2. Total path length
    total_len = 0.0
    for stroke in strokes:
        for i in range(1, len(stroke)):
            total_len += math.hypot(stroke[i][0] - stroke[i-1][0], stroke[i][1] - stroke[i-1][1])

    # 3. Start to End closure distance
    start_pt = strokes[0][0]
    end_pt = strokes[-1][-1]
    closure_dist = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
    closure_ratio = closure_dist / max(total_len, 1e-4)

    # 4. Centroid
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    # 5. Radial distances from centroid
    radii = [math.hypot(p[0] - cx, p[1] - cy) for p in all_pts]
    mean_r = float(np.mean(radii)) if radii else 0.0
    std_r = float(np.std(radii)) if radii else 0.0
    radial_cv = (std_r / max(mean_r, 1e-4))  # Very low (<0.18) for circles!

    # 6. Straight line regression RMSE & Linearity
    try:
        if len(xs) > 2 and width > 0.05:
            slope, intercept = np.polyfit(xs, ys, 1)
            residuals = [abs(y - (slope * x + intercept)) for x, y in zip(xs, ys)]
            rmse = float(np.mean(residuals))
            linearity = max(0.0, 1.0 - rmse * 10)
        else:
            linearity = 1.0 if width < 0.05 and height > 0.2 else 0.0
    except Exception:
        linearity = 0.0

    # 7. Sharp corner detection (>45 degrees)
    corners = 0
    for stroke in strokes:
        if len(stroke) < 5:
            continue
        step = max(1, len(stroke) // 16)
        for i in range(step, len(stroke) - step, step):
            p_prev = stroke[i - step]
            p_curr = stroke[i]
            p_next = stroke[min(len(stroke) - 1, i + step)]
            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag = math.hypot(*v1) * math.hypot(*v2)
            if mag > 1e-5:
                cos_ang = np.clip(dot / mag, -1.0, 1.0)
                angle_deg = math.degrees(math.acos(cos_ang))
                if angle_deg > 45:
                    corners += 1

    # 8. Quadrant mass distribution (top-left, top-right, bottom-left, bottom-right)
    q_tl = sum(1 for p in all_pts if p[0] <= cx and p[1] <= cy) / len(all_pts)
    q_tr = sum(1 for p in all_pts if p[0] > cx and p[1] <= cy) / len(all_pts)
    q_bl = sum(1 for p in all_pts if p[0] <= cx and p[1] > cy) / len(all_pts)
    q_br = sum(1 for p in all_pts if p[0] > cx and p[1] > cy) / len(all_pts)

    # 9. Top vs Bottom mass (important for 'A' where top is a point, bottom is wide)
    top_mass = sum(1 for p in all_pts if p[1] < min_y + height * 0.4) / len(all_pts)
    bottom_mass = sum(1 for p in all_pts if p[1] > min_y + height * 0.6) / len(all_pts)

    # 10. Horizontal crossing count at middle height
    mid_y = min_y + height * 0.5
    crossings = 0
    for stroke in strokes:
        for i in range(1, len(stroke)):
            y1, y2 = stroke[i-1][1], stroke[i][1]
            if (y1 <= mid_y < y2) or (y2 <= mid_y < y1):
                crossings += 1

    stroke_count = len(strokes)

    features = [
        aspect_ratio,
        total_len,
        closure_ratio,
        cx,
        cy,
        mean_r,
        std_r,
        radial_cv,
        linearity,
        float(corners),
        float(stroke_count),
        q_tl,
        q_tr,
        top_mass,
        bottom_mass,
        float(crossings)
    ]
    return np.array(features, dtype=np.float32)


def extract_combined_features(strokes: List[List[Any]]) -> np.ndarray:
    """
    Extract complete 272-dimensional vector: 256 raster features + 16 geometric features.
    """
    norm_strokes = normalize_strokes(strokes)
    raster = rasterize_strokes_to_grid(norm_strokes, grid_size=16)
    geom = extract_geometric_features(norm_strokes)
    return np.concatenate([raster, geom])


# =====================================================================
# DATASET GENERATORS & MODEL TRAINERS
# =====================================================================

def generate_shape_strokes(shape_type: str, jitter: float = 0.02, scale_x: float = 1.0, scale_y: float = 1.0) -> List[List[Tuple[float, float]]]:
    """Generate stroke points for basic geometric shapes with realistic hand tremor/jitter."""
    pts = []
    np_rand = np.random.normal

    if shape_type == "triangle":
        # 3 vertices: top (0.5, 0.1), bottom-right (0.9, 0.9), bottom-left (0.1, 0.9)
        v = [(0.5 * scale_x, 0.1 * scale_y), (0.9 * scale_x, 0.9 * scale_y), (0.1 * scale_x, 0.9 * scale_y), (0.5 * scale_x, 0.1 * scale_y)]
        for i in range(3):
            for t in np.linspace(0, 1, 15):
                x = v[i][0] + t * (v[i+1][0] - v[i][0]) + np_rand(0, jitter)
                y = v[i][1] + t * (v[i+1][1] - v[i][1]) + np_rand(0, jitter)
                pts.append((x, y))
        return [pts]

    elif shape_type == "square":
        # 4 vertices: (0.15, 0.15) -> (0.85, 0.15) -> (0.85, 0.85) -> (0.15, 0.85) -> close
        v = [
            (0.15 * scale_x, 0.15 * scale_y),
            (0.85 * scale_x, 0.15 * scale_y),
            (0.85 * scale_x, 0.85 * scale_y),
            (0.15 * scale_x, 0.85 * scale_y),
            (0.15 * scale_x, 0.15 * scale_y)
        ]
        for i in range(4):
            for t in np.linspace(0, 1, 12):
                x = v[i][0] + t * (v[i+1][0] - v[i][0]) + np_rand(0, jitter)
                y = v[i][1] + t * (v[i+1][1] - v[i][1]) + np_rand(0, jitter)
                pts.append((x, y))
        return [pts]

    elif shape_type == "rectangle":
        # Wider than tall or taller than wide
        sx = scale_x * 1.5
        sy = scale_y * 0.8
        v = [
            (0.1 * sx, 0.2 * sy),
            (0.9 * sx, 0.2 * sy),
            (0.9 * sx, 0.8 * sy),
            (0.1 * sx, 0.8 * sy),
            (0.1 * sx, 0.2 * sy)
        ]
        for i in range(4):
            for t in np.linspace(0, 1, 12):
                x = v[i][0] + t * (v[i+1][0] - v[i][0]) + np_rand(0, jitter)
                y = v[i][1] + t * (v[i+1][1] - v[i][1]) + np_rand(0, jitter)
                pts.append((x, y))
        return [pts]

    elif shape_type == "circle":
        cx, cy, r = 0.5 * scale_x, 0.5 * scale_y, 0.38 * min(scale_x, scale_y)
        for theta in np.linspace(0, 2 * math.pi, 45):
            x = cx + r * math.cos(theta) + np_rand(0, jitter)
            y = cy + r * math.sin(theta) + np_rand(0, jitter)
            pts.append((x, y))
        return [pts]

    elif shape_type == "line":
        # Straight line across or diagonal
        p0 = (0.1 * scale_x, 0.5 * scale_y)
        p1 = (0.9 * scale_x, 0.5 * scale_y)
        for t in np.linspace(0, 1, 30):
            x = p0[0] + t * (p1[0] - p0[0]) + np_rand(0, jitter * 0.5)
            y = p0[1] + t * (p1[1] - p0[1]) + np_rand(0, jitter * 0.5)
            pts.append((x, y))
        return [pts]

    return [pts]


def generate_letter_strokes(letter: str, jitter: float = 0.02) -> List[List[Tuple[float, float]]]:
    """Generate realistic stroke sequences for letters like A, B, C, D, P, R, etc."""
    strokes = []
    np_rand = np.random.normal

    if letter == "A":
        # Stroke 1: Left slant up (0.2, 0.9) -> (0.5, 0.1)
        s1 = [(0.2 + t * 0.3 + np_rand(0, jitter), 0.9 - t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        # Stroke 2: Right slant down (0.5, 0.1) -> (0.8, 0.9)
        s2 = [(0.5 + t * 0.3 + np_rand(0, jitter), 0.1 + t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        # Stroke 3: Cross bar (0.32, 0.55) -> (0.68, 0.55)
        s3 = [(0.32 + t * 0.36 + np_rand(0, jitter), 0.55 + np_rand(0, jitter)) for t in np.linspace(0, 1, 10)]
        strokes = [s1, s2, s3]

    elif letter == "B":
        # Stroke 1: Vertical spine (0.25, 0.1) -> (0.25, 0.9)
        s1 = [(0.25 + np_rand(0, jitter), 0.1 + t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        # Stroke 2: Top loop (0.25, 0.1) -> (0.65, 0.3) -> (0.25, 0.5)
        s2 = []
        for theta in np.linspace(-math.pi/2, math.pi/2, 16):
            s2.append((0.25 + 0.35 * math.cos(theta) + np_rand(0, jitter), 0.3 + 0.2 * math.sin(theta) + np_rand(0, jitter)))
        # Stroke 3: Bottom loop (0.25, 0.5) -> (0.7, 0.7) -> (0.25, 0.9)
        s3 = []
        for theta in np.linspace(-math.pi/2, math.pi/2, 16):
            s3.append((0.25 + 0.4 * math.cos(theta) + np_rand(0, jitter), 0.7 + 0.2 * math.sin(theta) + np_rand(0, jitter)))
        strokes = [s1, s2, s3]

    elif letter == "P":
        # Vertical spine + top loop only
        s1 = [(0.25 + np_rand(0, jitter), 0.1 + t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        s2 = []
        for theta in np.linspace(-math.pi/2, math.pi/2, 18):
            s2.append((0.25 + 0.4 * math.cos(theta) + np_rand(0, jitter), 0.32 + 0.22 * math.sin(theta) + np_rand(0, jitter)))
        strokes = [s1, s2]

    elif letter == "R":
        # Vertical spine + top loop + diagonal leg
        s1 = [(0.25 + np_rand(0, jitter), 0.1 + t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        s2 = []
        for theta in np.linspace(-math.pi/2, math.pi/2, 18):
            s2.append((0.25 + 0.4 * math.cos(theta) + np_rand(0, jitter), 0.32 + 0.22 * math.sin(theta) + np_rand(0, jitter)))
        s3 = [(0.4 + t * 0.4 + np_rand(0, jitter), 0.54 + t * 0.36 + np_rand(0, jitter)) for t in np.linspace(0, 1, 12)]
        strokes = [s1, s2, s3]

    elif letter == "C":
        # Open curve: angle from pi/4 to 7pi/4
        s1 = []
        for theta in np.linspace(0.8, 2 * math.pi - 0.8, 30):
            s1.append((0.55 + 0.38 * math.cos(theta) + np_rand(0, jitter), 0.5 - 0.38 * math.sin(theta) + np_rand(0, jitter)))
        strokes = [s1]

    elif letter == "D":
        # Vertical spine + single large loop
        s1 = [(0.25 + np_rand(0, jitter), 0.1 + t * 0.8 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        s2 = []
        for theta in np.linspace(-math.pi/2, math.pi/2, 25):
            s2.append((0.25 + 0.55 * math.cos(theta) + np_rand(0, jitter), 0.5 + 0.4 * math.sin(theta) + np_rand(0, jitter)))
        strokes = [s1, s2]

    elif letter == "O":
        s1 = []
        for theta in np.linspace(0, 2 * math.pi, 36):
            s1.append((0.5 + 0.35 * math.cos(theta) + np_rand(0, jitter), 0.5 + 0.38 * math.sin(theta) + np_rand(0, jitter)))
        strokes = [s1]

    elif letter == "S":
        # S-curve
        s1 = []
        for t in np.linspace(0, 1, 30):
            x = 0.5 + 0.25 * math.sin(t * 2 * math.pi) + np_rand(0, jitter)
            y = 0.15 + t * 0.7 + np_rand(0, jitter)
            s1.append((x, y))
        strokes = [s1]

    elif letter == "T":
        # Top bar + vertical stem
        s1 = [(0.15 + t * 0.7 + np_rand(0, jitter), 0.15 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        s2 = [(0.5 + np_rand(0, jitter), 0.15 + t * 0.75 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        strokes = [s1, s2]

    elif letter == "L":
        # Vertical stem + bottom bar
        s1 = [(0.25 + np_rand(0, jitter), 0.15 + t * 0.7 + np_rand(0, jitter)) for t in np.linspace(0, 1, 15)]
        s2 = [(0.25 + t * 0.55 + np_rand(0, jitter), 0.85 + np_rand(0, jitter)) for t in np.linspace(0, 1, 12)]
        strokes = [s1, s2]

    else:
        # Generic stroke
        strokes = [[(0.2 + t * 0.6, 0.2 + t * 0.6) for t in np.linspace(0, 1, 20)]]

    return strokes


def train_and_save_models():
    """
    Generate synthetic dataset, train MLP neural classifiers, and save joblib models.
    """
    print("[*] Generating synthetic training samples for geometric shapes...")
    X_shapes = []
    y_shapes = []

    for shape in SHAPE_CLASSES:
        for _ in range(120):  # 120 variations per shape
            jitter = float(np.random.uniform(0.005, 0.035))
            scale_x = float(np.random.uniform(0.8, 1.2))
            scale_y = float(np.random.uniform(0.8, 1.2))
            strokes = generate_shape_strokes(shape, jitter=jitter, scale_x=scale_x, scale_y=scale_y)
            feat = extract_combined_features(strokes)
            X_shapes.append(feat)
            y_shapes.append(shape)

    X_shapes = np.array(X_shapes, dtype=np.float32)
    y_shapes = np.array(y_shapes)

    print(f"[*] Training shape MLPClassifier on {len(X_shapes)} samples...")
    shape_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=400,
        random_state=42,
        alpha=0.001
    )
    shape_clf.fit(X_shapes, y_shapes)
    joblib.dump(shape_clf, SHAPE_MODEL_PATH)
    print(f"[+] Shape model saved to {SHAPE_MODEL_PATH} ({os.path.getsize(SHAPE_MODEL_PATH) / 1024:.1f} KB)")

    print("[*] Generating synthetic training samples for letters...")
    X_letters = []
    y_letters = []

    for letter in LETTER_CLASSES:
        for _ in range(100):  # 100 variations per letter
            jitter = float(np.random.uniform(0.005, 0.035))
            strokes = generate_letter_strokes(letter, jitter=jitter)
            feat = extract_combined_features(strokes)
            X_letters.append(feat)
            y_letters.append(letter)

    X_letters = np.array(X_letters, dtype=np.float32)
    y_letters = np.array(y_letters)

    print(f"[*] Training letter MLPClassifier on {len(X_letters)} samples...")
    letter_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=400,
        random_state=42,
        alpha=0.001
    )
    letter_clf.fit(X_letters, y_letters)
    joblib.dump(letter_clf, LETTER_MODEL_PATH)
    print(f"[+] Letter model saved to {LETTER_MODEL_PATH} ({os.path.getsize(LETTER_MODEL_PATH) / 1024:.1f} KB)")


# Global cached model instances
_shape_model = None
_letter_model = None


def get_shape_model():
    global _shape_model
    if _shape_model is None:
        if not os.path.exists(SHAPE_MODEL_PATH):
            train_and_save_models()
        _shape_model = joblib.load(SHAPE_MODEL_PATH)
    return _shape_model


def get_letter_model():
    global _letter_model
    if _letter_model is None:
        if not os.path.exists(LETTER_MODEL_PATH):
            train_and_save_models()
        _letter_model = joblib.load(LETTER_MODEL_PATH)
    return _letter_model


def rasterize_strokes_to_96x96(strokes: List[Any]) -> np.ndarray:
    """
    Rasterize strokes onto a 96x96 canvas for EMNIST MobileNetV2 ONNX model.
    EMNIST conventions: black background (0.0), white ink (255.0).
    """
    grid = np.zeros((96, 96), dtype=np.float32)
    all_pts = []
    for s in strokes:
        curr = extract_points_from_stroke(s)
        all_pts.extend(curr)

    if not all_pts:
        return grid

    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    w = max(1.0, max_x - min_x)
    h = max(1.0, max_y - min_y)
    max_dim = max(w, h)

    for s in strokes:
        pts = extract_points_from_stroke(s)
        if not pts:
            continue
        norm_pts = []
        for x, y in pts:
            nx = int(np.clip(16 + 64 * ((x - min_x) / max_dim), 0, 95))
            ny = int(np.clip(16 + 64 * ((y - min_y) / max_dim), 0, 95))
            norm_pts.append((nx, ny))

        for i in range(len(norm_pts)):
            x0, y0 = norm_pts[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    grid[int(np.clip(y0 + dy, 0, 95)), int(np.clip(x0 + dx, 0, 95))] = 255.0
            if i > 0:
                x_prev, y_prev = norm_pts[i - 1]
                dist = math.hypot(x0 - x_prev, y0 - y_prev)
                steps = max(1, int(dist * 2))
                for st in range(steps):
                    ix = int(np.clip(x_prev + (st / steps) * (x0 - x_prev), 0, 95))
                    iy = int(np.clip(y_prev + (st / steps) * (y0 - y_prev), 0, 95))
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            grid[int(np.clip(iy + dy, 0, 95)), int(np.clip(ix + dx, 0, 95))] = 255.0
    return grid


def predict_drawing(strokes: List[List[Any]], target_type: str = "shape", target: str = None) -> Dict[str, Any]:
    """
    Main prediction entry point for drawings from the front-end canvas.
    Returns recognized label, confidence percentage, probabilities, and kinematics.
    Uses pure client-side geometry for shapes, and an ensemble of MobileNetV2 ONNX
    + Scikit-Learn MLP for letters.
    """
    if not strokes or len(strokes) == 0:
        return {
            "status": "empty",
            "recognized": "None",
            "confidence": 0,
            "is_match": False,
            "probabilities": {},
            "kinematics": {"smoothness": 0.85, "straightness": 0.85, "tremor_stability": 0.85}
        }

    # Normalize strokes format:
    # Only wrap if strokes is a flat list of coordinate points, NOT a list of stroke objects
    if isinstance(strokes, list) and len(strokes) > 0:
        first = strokes[0]
        # Flat list of point dicts: [{'x': 100, 'y': 150}, ...]
        if isinstance(first, dict) and ('x' in first or 'y' in first) and 'points' not in first:
            strokes = [strokes]
        # Flat list of numeric coordinate pairs/triplets: [[100, 150], [110, 150]]
        elif isinstance(first, (list, tuple)) and len(first) in (2, 3) and all(isinstance(v, (int, float)) for v in first[:2]):
            strokes = [strokes]

    # Handle positional target e.g. predict_drawing(strokes, 'line')
    if target is None and target_type.lower() in [s.lower() for s in SHAPE_CLASSES]:
        target = target_type
        target_type = "shape"
    elif target is None and len(target_type) == 1 and target_type.isalpha():
        target = target_type.upper()
        target_type = "letter"

    feat = extract_combined_features(strokes).reshape(1, -1)
    norm_strokes = normalize_strokes(strokes)
    geom = extract_geometric_features(norm_strokes)

    # Compute kinematics
    # geom indices: 0:aspect_ratio, 7:radial_cv, 8:linearity, 9:corners
    linearity = float(geom[8])
    radial_cv = float(geom[7])
    corners = int(geom[9])

    kinematics = {
        "smoothness": round(min(1.0, max(0.4, 0.7 + linearity * 0.25)), 2),
        "straightness": round(min(1.0, max(0.3, linearity)), 2),
        "tremor_stability": round(min(1.0, max(0.5, 1.0 - radial_cv * 0.5)), 2),
        "corners": corners,
        "circularity": round(float(np.clip(1.0 - radial_cv, 0.0, 1.0)), 2)
    }

    if target_type.lower() == "shape":
        model = get_shape_model()
        probs = model.predict_proba(feat)[0]
        classes = model.classes_
        top_idx = int(np.argmax(probs))
        recognized = str(classes[top_idx])
        top_prob = float(probs[top_idx])

        # If user is drawing a straight line or linearity is very high:
        target_name = (target or "").lower()
        
        # Line Tracing specific path deviation verification
        if target_name == "line":
            raw_pts = []
            for s in strokes:
                raw_pts.extend(extract_points_from_stroke(s))
            
            if raw_pts:
                y_diffs = [p[1] - 150.0 for p in raw_pts]
                y_rmse = math.sqrt(sum(d * d for d in y_diffs) / len(y_diffs))
                all_y = [p[1] for p in raw_pts]
                all_x = [p[0] for p in raw_pts]
                y_span = max(all_y) - min(all_y)
                x_span = max(all_x) - min(all_x)
                kinematics["rmse"] = round(y_rmse, 1)
                kinematics["x_span"] = round(x_span, 1)

                if y_rmse > 20.0 or y_span > 45.0 or linearity < 0.65 or x_span < 60.0:
                    # User drew away from dotted line or drew diagonal / curved line or insufficient progress
                    recognized = "Off Path Deviation"
                    top_prob = max(0.15, min(0.42, 1.0 - (y_rmse / 50.0)))
                    is_match = False
                else:
                    recognized = "line"
                    top_prob = max(0.70, min(0.98, 1.0 - (y_rmse / 35.0)))
                    is_match = True
            else:
                recognized = "line" if linearity > 0.88 else str(classes[top_idx])
                is_match = (recognized.lower() == target_name)
        else:
            if linearity > 0.88:
                recognized = "line"
                top_prob = max(top_prob, 0.92)
            prob_dict = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
            is_match = (recognized.lower() == target_name) if target_name else (top_prob >= 0.50)

        prob_dict = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
        shape_title = recognized.capitalize()
        return {
            "status": "success",
            "engine": "sklearn_mlp_geom",
            "recognized": shape_title,
            "confidence": int(round(top_prob * 100)),
            "is_match": bool(is_match),
            "probabilities": prob_dict,
            "kinematics": kinematics
        }

    else:
        # Alphabet recognition: Ensemble between MobileNetV2 ONNX Model (EMNIST) and Scikit-Learn MLP
        target_char = (target or "").upper().strip()

        # 1. Scikit-Learn MLP model (trained on stroke dynamics & 16x16 density)
        mlp_model = get_letter_model()
        mlp_probs = mlp_model.predict_proba(feat)[0]
        mlp_classes = [str(c).upper() for c in mlp_model.classes_]
        mlp_dict = {c: float(p) for c, p in zip(mlp_classes, mlp_probs)}
        mlp_top_idx = int(np.argmax(mlp_probs))
        mlp_pred = mlp_classes[mlp_top_idx]
        mlp_conf = float(mlp_probs[mlp_top_idx])

        # 2. ONNX MobileNetV2 model (EMNIST character classifier)
        onnx_sess = get_onnx_letter_session()
        onnx_pred = None
        onnx_conf = 0.0
        onnx_dict = {}

        if onnx_sess is not None:
            try:
                img_96 = rasterize_strokes_to_96x96(strokes)
                x_input = ((img_96 / 255.0 - 0.5) / 0.5)[np.newaxis, np.newaxis, :, :].astype(np.float32)
                logits = onnx_sess.run(None, {"input": x_input})[0][0]
                exp_logits = np.exp(logits - np.max(logits))
                onnx_probs = exp_logits / np.sum(exp_logits)
                onnx_top_idx = int(np.argmax(onnx_probs))
                onnx_pred = ONNX_ALPHA_CLASSES[onnx_top_idx]
                onnx_conf = float(onnx_probs[onnx_top_idx])
                onnx_dict = {c: float(p) for c, p in zip(ONNX_ALPHA_CLASSES, onnx_probs)}
            except Exception as e:
                print(f"[!] ONNX inference warning: {e}")

        # Combined probabilities
        all_chars = set(list(mlp_dict.keys()) + list(onnx_dict.keys()))
        combined_probs = {}
        for c in all_chars:
            if c != "blank":
                p_m = mlp_dict.get(c, 0.0)
                p_o = onnx_dict.get(c, 0.0)
                combined_probs[c] = round(max(p_m, p_o) * 100, 1)

        # Match evaluation
        if target_char:
            mlp_target_prob = mlp_dict.get(target_char, 0.0)
            onnx_target_prob = onnx_dict.get(target_char, 0.0)
            # Letter matches if either model predicts it or has significant confidence
            is_match = (
                (mlp_pred == target_char) or
                (onnx_pred == target_char) or
                (mlp_target_prob >= 0.25) or
                (onnx_target_prob >= 0.15)
            )
            if is_match:
                recognized = target_char
                best_prob = max(mlp_target_prob, onnx_target_prob, mlp_conf if mlp_pred == target_char else 0.0)
                confidence = int(round(max(best_prob, 0.75) * 100))
            else:
                recognized = mlp_pred if mlp_conf >= onnx_conf else (onnx_pred or mlp_pred)
                confidence = int(round(max(mlp_conf, onnx_conf) * 100))
        else:
            # Open classification
            if mlp_pred == onnx_pred:
                recognized = mlp_pred
                confidence = int(round(max(mlp_conf, onnx_conf) * 100))
            elif mlp_conf >= onnx_conf:
                recognized = mlp_pred
                confidence = int(round(mlp_conf * 100))
            else:
                recognized = onnx_pred or mlp_pred
                confidence = int(round(onnx_conf * 100))
            is_match = (confidence >= 40 and recognized != "blank")

        return {
            "status": "success",
            "engine": "emnist_onnx_mlp_ensemble",
            "recognized": recognized.upper(),
            "confidence": confidence,
            "is_match": bool(is_match),
            "probabilities": combined_probs,
            "kinematics": kinematics
        }


if __name__ == "__main__":
    train_and_save_models()
    print("\n--- Testing Model Predictions ---")
    tri_strokes = generate_shape_strokes("triangle", jitter=0.01)
    res_tri = predict_drawing(tri_strokes, "shape", "Triangle")
    print("Triangle test:", res_tri)

    sq_strokes = generate_shape_strokes("square", jitter=0.01)
    res_sq = predict_drawing(sq_strokes, "shape", "Square")
    print("Square test:", res_sq)

    circ_strokes = generate_shape_strokes("circle", jitter=0.01)
    res_circ = predict_drawing(circ_strokes, "shape", "Circle")
    print("Circle test:", res_circ)

    a_strokes = generate_letter_strokes("A", jitter=0.01)
    res_a = predict_drawing(a_strokes, "letter", "A")
    print("Letter A test:", res_a)

    b_strokes = generate_letter_strokes("B", jitter=0.01)
    res_b = predict_drawing(b_strokes, "letter", "B")
    print("Letter B test:", res_b)

    c_strokes = generate_letter_strokes("C", jitter=0.01)
    res_c = predict_drawing(c_strokes, "letter", "C")
    print("Letter C test:", res_c)
