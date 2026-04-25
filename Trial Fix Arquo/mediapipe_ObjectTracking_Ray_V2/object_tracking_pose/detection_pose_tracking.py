"""
MediaPipe Object Detection + Pose Estimation - Ultra-Stable Tracking v3
========================================================================
Menggabungkan:
1. Object Detection (EfficientDet-Lite0) dengan Kalman Filter tracking
2. Pose Estimation (MediaPipe Pose) untuk deteksi skeleton tubuh

Optimasi FPS:
- Object Detection: LIVE_STREAM mode (async, non-blocking)
- Pose Detection: model_complexity=0 (lightest model)
- Pose bisa dijalankan setiap N frame (POSE_INTERVAL)
- Frame di-downscale untuk pose input (POSE_SCALE_FACTOR)

Model: EfficientDet-Lite0 (COCO 80 classes) + MediaPipe Pose
Framework: MediaPipe Tasks API + MediaPipe Solutions + OpenCV
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
from collections import deque

# ============================================================
# KONFIGURASI UMUM
# ============================================================
MODEL_PATH = "efficientdet_lite0.tflite"
MAX_RESULTS = 10           # Maksimal objek yang dideteksi
SCORE_THRESHOLD = 0.20     # Threshold confidence object detection
CAMERA_INDEX = 1           # Index kamera (0 = default webcam)
FRAME_WIDTH = 1280         # Lebar frame kamera
FRAME_HEIGHT = 720         # Tinggi frame kamera

# ============================================================
# KONFIGURASI POSE ESTIMATION
# ============================================================
ENABLE_POSE = True              # Toggle pose detection ON/OFF (tekan 'p')
POSE_MODEL_PATH = "pose_landmarker_lite.task"  # Model pose (auto-download)
POSE_MIN_DETECTION_CONF = 0.5   # Minimum confidence untuk deteksi pose awal
POSE_MIN_TRACKING_CONF = 0.5    # Minimum confidence untuk tracking pose
POSE_INTERVAL = 1               # Jalankan pose setiap N frame (1=setiap frame, 2=skip 1)
POSE_SCALE_FACTOR = 1.0         # Downscale factor untuk pose input (0.5 = half res)
POSE_DRAW_STYLE = "full"        # "full", "minimal", "skeleton_only"

# Warna untuk pose
POSE_LANDMARK_COLOR = (0, 255, 128)     # Hijau terang untuk landmark
POSE_CONNECTION_COLOR = (230, 216, 173)  # Light blue untuk connections
POSE_LANDMARK_SIZE = 4                   # Ukuran titik landmark
POSE_CONNECTION_THICKNESS = 2            # Ketebalan garis koneksi

# Pose connections (33 landmarks) - didefinisikan manual karena mp.solutions tidak tersedia
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
])

# ============================================================
# KONFIGURASI TRACKING & STABILISASI (sama dengan v2)
# ============================================================
MAX_CENTROID_DISTANCE = 200
SIZE_WEIGHT = 0.3
PROCESS_NOISE = 1e-2
MEASUREMENT_NOISE = 1e-1
MAX_MISSING_FRAMES = 30
MIN_HIT_STREAK = 2
DISPLAY_THRESHOLD = 0.18
BBOX_SMOOTHING = 0.3
SCORE_SMOOTHING = 0.3
DETECTION_INTERVAL = 1
USE_MULTI_SCALE = False

# Warna untuk bounding box (BGR format)
COLORS = [
    (0, 255, 0),     # Hijau
    (255, 0, 0),     # Biru
    (0, 0, 255),     # Merah
    (255, 255, 0),   # Cyan
    (0, 255, 255),   # Kuning
    (255, 0, 255),   # Magenta
    (128, 255, 0),   # Lime
    (255, 128, 0),   # Light Blue
    (0, 128, 255),   # Orange
    (128, 0, 255),   # Purple
]

# Target objek yang ingin di-highlight (kosong = tampilkan semua)
TARGET_OBJECTS = []


# ============================================================
# KALMAN FILTER untuk Bounding Box Tracking
# ============================================================
class BBoxKalmanFilter:
    """
    Kalman Filter untuk tracking bounding box [cx, cy, w, h].
    State: [cx, cy, w, h, vx, vy, vw, vh] (posisi + velocity)
    """

    def __init__(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0

        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        self.F = np.eye(8)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.F[3, 7] = 1

        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.P = np.eye(8) * 10.0
        self.P[4, 4] = 100.0
        self.P[5, 5] = 100.0
        self.P[6, 6] = 100.0
        self.P[7, 7] = 100.0

        self.Q = np.eye(8) * PROCESS_NOISE
        self.Q[0, 0] = PROCESS_NOISE * 0.5
        self.Q[1, 1] = PROCESS_NOISE * 0.5
        self.Q[4, 4] = PROCESS_NOISE * 2
        self.Q[5, 5] = PROCESS_NOISE * 2

        self.R = np.eye(4) * MEASUREMENT_NOISE

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.state[2] = max(self.state[2], 10)
        self.state[3] = max(self.state[3], 10)
        return self.get_bbox()

    def update(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        z = np.array([cx, cy, w, h], dtype=np.float64)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y_residual = z - self.H @ self.state
        self.state = self.state + K @ y_residual

        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

        self.state[2] = max(self.state[2], 10)
        self.state[3] = max(self.state[3], 10)
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.state[:4]
        return [cx - w / 2.0, cy - h / 2.0, w, h]

    def get_centroid(self):
        return self.state[:2].copy()

    def get_velocity(self):
        return self.state[4:6].copy()

    def get_predicted_centroid(self, steps=1):
        cx, cy = self.state[0], self.state[1]
        vx, vy = self.state[4], self.state[5]
        return np.array([cx + vx * steps, cy + vy * steps])


# ============================================================
# TRACKED OBJECT CLASS
# ============================================================
class TrackedObject:
    """Represents a single tracked object with Kalman filter smoothing."""

    _next_id = 1

    def __init__(self, bbox, category_name, score):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1

        self.category_name = category_name
        self.kf = BBoxKalmanFilter(bbox)
        self.raw_bbox = list(bbox)
        self.smooth_bbox = list(bbox)
        self.score = score
        self.smooth_score = score
        self.peak_score = score
        self.score_history = deque(maxlen=30)
        self.score_history.append(score)
        self.hit_streak = 1
        self.missing_frames = 0
        self.total_hits = 1
        self.age = 1
        self.last_size = (bbox[2], bbox[3])

    def predict(self):
        predicted_bbox = self.kf.predict()
        return predicted_bbox

    def update(self, bbox, score):
        self.raw_bbox = list(bbox)
        self.score = score
        kf_bbox = self.kf.update(bbox)

        alpha = BBOX_SMOOTHING
        for i in range(4):
            self.smooth_bbox[i] = alpha * kf_bbox[i] + (1 - alpha) * self.smooth_bbox[i]

        self.smooth_score = (
            SCORE_SMOOTHING * score +
            (1 - SCORE_SMOOTHING) * self.smooth_score
        )
        self.peak_score = max(self.peak_score, score)
        self.score_history.append(score)
        self.hit_streak += 1
        self.missing_frames = 0
        self.total_hits += 1
        self.age += 1
        self.last_size = (bbox[2], bbox[3])

    def mark_missing(self):
        self.hit_streak = 0
        self.missing_frames += 1
        self.age += 1
        predicted_bbox = self.kf.get_bbox()
        alpha = 0.2
        for i in range(4):
            self.smooth_bbox[i] = alpha * predicted_bbox[i] + (1 - alpha) * self.smooth_bbox[i]
        fade_rate = 0.92 if self.total_hits > 5 else 0.85
        self.smooth_score *= fade_rate

    def should_display(self):
        if self.total_hits < MIN_HIT_STREAK:
            return False
        if self.smooth_score < DISPLAY_THRESHOLD:
            return False
        if self.missing_frames > MAX_MISSING_FRAMES:
            return False
        return True

    def is_expired(self):
        grace = MAX_MISSING_FRAMES
        if self.total_hits > 10:
            grace = int(MAX_MISSING_FRAMES * 1.5)
        if self.total_hits > 30:
            grace = MAX_MISSING_FRAMES * 2
        return self.missing_frames > grace

    def get_display_bbox(self):
        return [int(round(v)) for v in self.smooth_bbox]

    def get_opacity(self):
        if self.missing_frames == 0:
            return 1.0
        max_mf = MAX_MISSING_FRAMES
        return max(0.3, 1.0 - (self.missing_frames / max_mf) * 0.7)

    def get_centroid(self):
        return self.kf.get_centroid()

    def get_predicted_centroid(self, steps=1):
        return self.kf.get_predicted_centroid(steps)


# ============================================================
# OBJECT TRACKER with Centroid + Size Distance Matching
# ============================================================
class ObjectTracker:
    """Multi-object tracker dengan Kalman Filter + Centroid matching."""

    def __init__(self):
        self.tracked_objects = []

    @staticmethod
    def compute_matching_cost(tracked_obj, det_bbox, det_category):
        if tracked_obj.category_name != det_category:
            return float('inf')

        dx, dy, dw, dh = det_bbox
        det_cx = dx + dw / 2.0
        det_cy = dy + dh / 2.0

        pred_centroid = tracked_obj.get_predicted_centroid(steps=1)
        pred_cx, pred_cy = pred_centroid

        dist = np.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2)

        if dist > MAX_CENTROID_DISTANCE:
            return float('inf')

        tw, th = tracked_obj.last_size
        size_ratio_w = min(dw, tw) / max(dw, tw) if max(dw, tw) > 0 else 0
        size_ratio_h = min(dh, th) / max(dh, th) if max(dh, th) > 0 else 0
        size_sim = (size_ratio_w + size_ratio_h) / 2.0

        norm_dist = dist / MAX_CENTROID_DISTANCE
        size_penalty = (1.0 - size_sim) * SIZE_WEIGHT

        cost = norm_dist + size_penalty
        return cost

    @staticmethod
    def compute_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        for obj in self.tracked_objects:
            obj.predict()

        if not detections and not self.tracked_objects:
            return []

        if not detections:
            for obj in self.tracked_objects:
                obj.mark_missing()
            self.tracked_objects = [
                obj for obj in self.tracked_objects if not obj.is_expired()
            ]
            return [obj for obj in self.tracked_objects if obj.should_display()]

        if not self.tracked_objects:
            for bbox, cat, score in detections:
                self.tracked_objects.append(TrackedObject(bbox, cat, score))
            return [obj for obj in self.tracked_objects if obj.should_display()]

        num_trackers = len(self.tracked_objects)
        num_detections = len(detections)
        cost_matrix = np.full((num_trackers, num_detections), float('inf'))

        for t, obj in enumerate(self.tracked_objects):
            for d, (bbox, cat, score) in enumerate(detections):
                cost_matrix[t, d] = self.compute_matching_cost(obj, bbox, cat)

        matched_trackers = set()
        matched_detections = set()
        matches = []

        costs = []
        for t in range(num_trackers):
            for d in range(num_detections):
                if cost_matrix[t, d] < float('inf'):
                    costs.append((cost_matrix[t, d], t, d))
        costs.sort(key=lambda x: x[0])

        for cost, t_idx, d_idx in costs:
            if t_idx in matched_trackers or d_idx in matched_detections:
                continue
            matches.append((t_idx, d_idx))
            matched_trackers.add(t_idx)
            matched_detections.add(d_idx)

        for t_idx, d_idx in matches:
            bbox, cat, score = detections[d_idx]
            self.tracked_objects[t_idx].update(bbox, score)

        for t_idx in range(num_trackers):
            if t_idx not in matched_trackers:
                self.tracked_objects[t_idx].mark_missing()

        for d_idx in range(num_detections):
            if d_idx not in matched_detections:
                bbox, cat, score = detections[d_idx]
                self.tracked_objects.append(TrackedObject(bbox, cat, score))

        self.tracked_objects = [
            obj for obj in self.tracked_objects if not obj.is_expired()
        ]

        return [obj for obj in self.tracked_objects if obj.should_display()]


# ============================================================
# ASYNC DETECTOR (LIVE_STREAM mode - non-blocking)
# ============================================================
class AsyncDetector:
    """Detector menggunakan LIVE_STREAM mode (async, non-blocking)."""

    def __init__(self, model_path, max_results, score_threshold):
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._latest_result = None
        self._timestamp = 0

        def _callback(result, output_image, timestamp_ms):
            self._latest_result = result

        self.options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=_callback,
        )
        self.detector = ObjectDetector.create_from_options(self.options)

    def detect_async(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp += 1
        try:
            self.detector.detect_async(mp_image, self._timestamp)
        except Exception:
            pass

        detections = []
        if self._latest_result is not None:
            for detection in self._latest_result.detections:
                category_name = detection.categories[0].category_name
                score = detection.categories[0].score
                bbox = [
                    detection.bounding_box.origin_x,
                    detection.bounding_box.origin_y,
                    detection.bounding_box.width,
                    detection.bounding_box.height,
                ]
                detections.append((bbox, category_name, score))

        return detections

    def close(self):
        self.detector.close()


# ============================================================
# POSE DETECTOR - MediaPipe Tasks API (PoseLandmarker)
# ============================================================
class PoseDetector:
    """
    Wrapper untuk MediaPipe Tasks API PoseLandmarker.
    
    Menggunakan mp.tasks.vision.PoseLandmarker (bukan mp.solutions.pose
    yang sudah deprecated/dihapus di versi baru MediaPipe).
    
    Fitur:
    - Auto-download model pose_landmarker_lite.task
    - VIDEO running mode (sinkron, per-frame)
    - Optional downscaling untuk performa
    - Caching hasil untuk frame skipping
    """

    def __init__(self, model_path, min_detection_conf=0.5,
                 min_tracking_conf=0.5):
        # Download model jika belum ada
        self._download_pose_model(model_path)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            min_pose_presence_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Cache hasil terakhir
        self.last_landmarks = None  # list of NormalizedLandmark
        self.pose_detected = False
        self.frame_count = 0
        self._timestamp = 0

        # Performance tracking
        self.last_inference_time = 0

    def _download_pose_model(self, model_path):
        """Download pose landmarker model jika belum ada."""
        if os.path.exists(model_path):
            return

        print(f"[INFO] Mengunduh model pose: {model_path}...")
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        )

        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[INFO] Model pose berhasil diunduh: {model_path}")
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh model pose: {e}")
            print(f"[INFO] Pose detection akan dimatikan.")

    def detect(self, frame, skip=False):
        """
        Deteksi pose pada frame.
        
        Args:
            frame: BGR frame dari kamera
            skip: Jika True, gunakan hasil cache
            
        Returns:
            landmarks: list of NormalizedLandmark atau None
        """
        if skip and self.last_landmarks is not None:
            return self.last_landmarks

        self.frame_count += 1
        self._timestamp += 1

        # Optional downscale
        if POSE_SCALE_FACTOR < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * POSE_SCALE_FACTOR)
            new_h = int(h * POSE_SCALE_FACTOR)
            pose_frame = cv2.resize(frame, (new_w, new_h))
        else:
            pose_frame = frame

        # Convert BGR ke RGB
        rgb_frame = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Inference
        t_start = time.perf_counter()
        try:
            result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        except Exception:
            self.last_inference_time = (time.perf_counter() - t_start) * 1000
            self.pose_detected = False
            return None
        self.last_inference_time = (time.perf_counter() - t_start) * 1000  # ms

        # Check if any pose detected
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.last_landmarks = result.pose_landmarks[0]  # First person
            self.pose_detected = True
        else:
            self.pose_detected = False

        return self.last_landmarks

    def draw_pose(self, frame, landmarks=None):
        """
        Gambar pose landmarks dan koneksi pada frame.
        
        landmarks: list of NormalizedLandmark (dari Tasks API)
        """
        if landmarks is None:
            landmarks = self.last_landmarks

        if landmarks is None:
            return

        h, w = frame.shape[:2]

        if POSE_DRAW_STYLE == "full":
            self._draw_full_pose(frame, landmarks, w, h)
        elif POSE_DRAW_STYLE == "minimal":
            self._draw_minimal_pose(frame, landmarks, w, h)
        elif POSE_DRAW_STYLE == "skeleton_only":
            self._draw_skeleton_only(frame, landmarks, w, h)
        else:
            self._draw_full_pose(frame, landmarks, w, h)

    def _get_landmark_xy(self, landmark, w, h):
        """Extract (x, y) pixel coords from NormalizedLandmark."""
        return (int(landmark.x * w), int(landmark.y * h))

    def _get_visibility(self, landmark):
        """Get visibility (Tasks API uses 'visibility' attribute, may also have 'presence')."""
        vis = getattr(landmark, 'visibility', 0.0)
        return vis if vis is not None else 0.0

    def _draw_full_pose(self, frame, landmarks, w, h):
        """Draw semua landmark dan koneksi dengan style kustom."""
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            if self._get_visibility(start_lm) < 0.5 or self._get_visibility(end_lm) < 0.5:
                continue

            start_pt = self._get_landmark_xy(start_lm, w, h)
            end_pt = self._get_landmark_xy(end_lm, w, h)

            # Gradient-like effect
            cv2.line(frame, start_pt, end_pt, (100, 100, 100), POSE_CONNECTION_THICKNESS + 2)
            cv2.line(frame, start_pt, end_pt, POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)

        # Draw landmarks
        for idx, lm in enumerate(landmarks):
            if self._get_visibility(lm) < 0.5:
                continue

            cx, cy = self._get_landmark_xy(lm, w, h)

            # Ukuran berbeda berdasarkan bagian tubuh
            if idx in [0, 11, 12, 23, 24]:  # Nose, shoulders, hips
                size = POSE_LANDMARK_SIZE + 2
                color = (0, 255, 200)  # Aqua/teal
            elif idx in [15, 16, 19, 20, 27, 28, 31, 32]:  # Hands, feet
                size = POSE_LANDMARK_SIZE + 1
                color = (255, 200, 0)  # Kuning
            else:
                size = POSE_LANDMARK_SIZE
                color = POSE_LANDMARK_COLOR

            # Glow effect
            cv2.circle(frame, (cx, cy), size + 3, (50, 50, 50), -1)
            cv2.circle(frame, (cx, cy), size, color, -1)
            cv2.circle(frame, (cx, cy), size, (255, 255, 255), 1)

    def _draw_minimal_pose(self, frame, landmarks, w, h):
        """Draw hanya joint utama."""
        key_indices = [
            11, 12, 13, 14, 15, 16,  # Upper body
            23, 24, 25, 26, 27, 28,  # Lower body
        ]

        key_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
        ]

        for start_idx, end_idx in key_connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            if self._get_visibility(start_lm) < 0.5 or self._get_visibility(end_lm) < 0.5:
                continue

            start_pt = self._get_landmark_xy(start_lm, w, h)
            end_pt = self._get_landmark_xy(end_lm, w, h)
            cv2.line(frame, start_pt, end_pt, POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS + 1)

        for idx in key_indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            if self._get_visibility(lm) < 0.5:
                continue

            cx, cy = self._get_landmark_xy(lm, w, h)
            cv2.circle(frame, (cx, cy), POSE_LANDMARK_SIZE + 2, (50, 50, 50), -1)
            cv2.circle(frame, (cx, cy), POSE_LANDMARK_SIZE + 1, POSE_LANDMARK_COLOR, -1)

    def _draw_skeleton_only(self, frame, landmarks, w, h):
        """Draw hanya garis koneksi."""
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            if self._get_visibility(start_lm) < 0.5 or self._get_visibility(end_lm) < 0.5:
                continue

            start_pt = self._get_landmark_xy(start_lm, w, h)
            end_pt = self._get_landmark_xy(end_lm, w, h)
            cv2.line(frame, start_pt, end_pt, POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)

    def get_body_bbox(self, frame_shape):
        """Mendapatkan bounding box dari pose landmarks."""
        if self.last_landmarks is None:
            return None

        h, w = frame_shape[:2]
        xs = []
        ys = []
        for lm in self.last_landmarks:
            if self._get_visibility(lm) > 0.5:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

        if not xs:
            return None

        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))

        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def close(self):
        """Release resources."""
        self.landmarker.close()


# ============================================================
# TEMPORAL BUFFER
# ============================================================
class TemporalDetectionBuffer:
    """Buffer deteksi dari beberapa frame terakhir."""

    def __init__(self, buffer_size=3):
        self.buffer = deque(maxlen=buffer_size)

    def add_frame(self, detections):
        self.buffer.append(detections)

    def get_merged_detections(self, current_detections):
        if not self.buffer:
            return current_detections

        merged = list(current_detections)
        current_bboxes = [d[0] for d in current_detections]

        for frame_dets in self.buffer:
            for bbox, cat, score in frame_dets:
                is_duplicate = False
                for cur_bbox in current_bboxes:
                    iou = ObjectTracker.compute_iou(bbox, cur_bbox)
                    if iou > 0.3:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    merged.append((bbox, cat, score * 0.7))

        return merged


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def download_model():
    """Download model EfficientDet-Lite0 jika belum ada."""
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Model sudah ada: {MODEL_PATH}")
        return True

    print("[INFO] Mengunduh model EfficientDet-Lite0...")
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "object_detector/efficientdet_lite0/int8/latest/"
        "efficientdet_lite0.tflite"
    )

    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print(f"[INFO] Model berhasil diunduh: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh model: {e}")
        print(f"[INFO] Silakan unduh manual dari:\n  {model_url}")
        print(f"[INFO] Simpan sebagai: {MODEL_PATH}")
        return False


def get_color(index):
    return COLORS[index % len(COLORS)]


def draw_detection(frame, tracked_obj, color):
    """Draw bounding box and label for tracked object."""
    bbox = tracked_obj.get_display_bbox()
    x, y, w, h = bbox
    opacity = tracked_obj.get_opacity()

    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(10, min(w, fw - x))
    h = max(10, min(h, fh - y))

    adj_color = tuple(int(c * opacity) for c in color)
    thickness = 2

    cv2.rectangle(frame, (x, y), (x + w, y + h), adj_color, thickness)

    corner_len = min(30, max(8, w // 4), max(8, h // 4))
    corner_thick = max(3, int(4 * opacity))

    corners = [
        ((x, y), (x + corner_len, y), (x, y + corner_len)),
        ((x + w, y), (x + w - corner_len, y), (x + w, y + corner_len)),
        ((x, y + h), (x + corner_len, y + h), (x, y + h - corner_len)),
        ((x + w, y + h), (x + w - corner_len, y + h), (x + w, y + h - corner_len)),
    ]

    for corner, h_end, v_end in corners:
        cv2.line(frame, corner, h_end, adj_color, corner_thick)
        cv2.line(frame, corner, v_end, adj_color, corner_thick)

    label = f"#{tracked_obj.id} {tracked_obj.category_name}: {tracked_obj.smooth_score:.0%}"

    if tracked_obj.missing_frames > 0:
        label += f" [~{tracked_obj.missing_frames}]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_y = max(y - 10, text_h + 5)
    cv2.rectangle(
        frame,
        (x, label_y - text_h - 5),
        (x + text_w + 10, label_y + 5),
        adj_color,
        -1,
    )

    cv2.putText(
        frame, label, (x + 5, label_y),
        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
    )

    if tracked_obj.missing_frames == 0:
        vel = tracked_obj.kf.get_velocity()
        speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
        if speed > 2:
            centroid = tracked_obj.get_centroid()
            end_pt = centroid + vel * 5
            cv2.arrowedLine(
                frame,
                (int(centroid[0]), int(centroid[1])),
                (int(end_pt[0]), int(end_pt[1])),
                (0, 200, 255), 2, tipLength=0.3
            )


def draw_hud(frame, fps, detection_count, tracker_count, total_trackers,
             pose_enabled, pose_detected, pose_inference_ms):
    """Draw HUD with tracking + pose info."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay - sedikit lebih besar untuk pose info
    overlay = frame.copy()
    hud_height = 210 if pose_enabled else 175
    cv2.rectangle(overlay, (10, 10), (380, hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA
    )

    # Detection count
    cv2.putText(
        frame, f"Raw Detections: {detection_count}", (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Active display count
    cv2.putText(
        frame, f"Displayed Objects: {tracker_count}", (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA
    )

    # Total trackers
    cv2.putText(
        frame, f"Total Trackers: {total_trackers}", (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA
    )

    # Filter info
    if TARGET_OBJECTS:
        filter_text = f"Filter: {', '.join(TARGET_OBJECTS[:3])}"
        if len(TARGET_OBJECTS) > 3:
            filter_text += f" +{len(TARGET_OBJECTS) - 3}"
    else:
        filter_text = "Filter: Semua Objek"
    cv2.putText(
        frame, filter_text, (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
    )

    # Mode info
    cv2.putText(
        frame, "Kalman Filter + Centroid Tracking", (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
    )

    # Pose info
    if pose_enabled:
        pose_status = "DETECTED" if pose_detected else "No Person"
        pose_color = (0, 255, 200) if pose_detected else (100, 100, 100)
        cv2.putText(
            frame, f"Pose: {pose_status} ({pose_inference_ms:.1f}ms)", (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Pose Style: {POSE_DRAW_STYLE} | Model: Lite",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 180, 255), 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "Pose: OFF (tekan 'p' untuk ON)", (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
        )

    # Instructions
    cv2.putText(
        frame, "'q'/ESC=Keluar | 'p'=Pose ON/OFF | 's'=Style", (w - 420, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================
def main():
    """Main function - Object Detection + Pose Estimation."""
    global POSE_DRAW_STYLE

    print("=" * 65)
    print("  MediaPipe Object Detection + Pose Estimation")
    print("  Ultra-Stable Tracking v3")
    print("=" * 65)

    # Download model jika belum ada
    if not download_model():
        sys.exit(1)

    # Buka kamera
    print(f"\n[INFO] Membuka kamera (index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
#    cap = cv2.VideoCapture(r"C:\Users\Victus\Videos\Video Pokeb\Video Testing Dance.mp4")
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera!")
        print("[INFO] Coba ganti CAMERA_INDEX (0, 1, 2, ...)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolusi kamera: {actual_w}x{actual_h}")

    if TARGET_OBJECTS:
        print(f"[INFO] Filter objek: {', '.join(TARGET_OBJECTS)}")
    else:
        print("[INFO] Mendeteksi semua objek (tanpa filter)")

    print(f"\n[INFO] Konfigurasi:")
    print(f"  === Object Detection ===")
    print(f"  - Mode: LIVE_STREAM (Async) + Kalman Filter")
    print(f"  - Matching: Centroid Distance + Size (max={MAX_CENTROID_DISTANCE}px)")
    print(f"  - Grace Period: {MAX_MISSING_FRAMES} frames")
    print(f"  === Pose Estimation ===")
    print(f"  - Enabled: {ENABLE_POSE}")
    print(f"  - Model: {POSE_MODEL_PATH}")
    print(f"  - Interval: setiap {POSE_INTERVAL} frame")
    print(f"  - Scale Factor: {POSE_SCALE_FACTOR}")
    print(f"  - Draw Style: {POSE_DRAW_STYLE}")
    print(f"\n[INFO] Kontrol keyboard:")
    print(f"  'q' / ESC  = Keluar")
    print(f"  'p'        = Toggle Pose ON/OFF")
    print(f"  's'        = Ganti Pose Draw Style (full/minimal/skeleton)")
    print(f"\n[INFO] Deteksi dimulai!\n")

    # ---- Initialize Components ----

    # Async Object Detector
    detector = AsyncDetector(MODEL_PATH, MAX_RESULTS, SCORE_THRESHOLD)

    # Object Tracker
    tracker = ObjectTracker()

    # Temporal buffer
    temporal_buffer = TemporalDetectionBuffer(buffer_size=3)

    # Pose Detector
    pose_detector = PoseDetector(
        model_path=POSE_MODEL_PATH,
        min_detection_conf=POSE_MIN_DETECTION_CONF,
        min_tracking_conf=POSE_MIN_TRACKING_CONF,
    )

    # State
    pose_enabled = ENABLE_POSE
    pose_styles = ["full", "minimal", "skeleton_only"]
    pose_style_idx = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0

    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = time.time()

    # Category color mapping
    category_color_map = {}
    color_index = 0

    last_raw_detections = []
    pose_frame_counter = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Gagal membaca frame dari kamera.")
                break

            # Flip horizontal (mirror effect)
            frame = cv2.flip(frame, 1)

            # ========================================
            # 1. OBJECT DETECTION (Async - Non-blocking)
            # ========================================
            raw_detections = detector.detect_async(frame)

            if TARGET_OBJECTS:
                raw_detections = [
                    (bbox, cat, score) for bbox, cat, score in raw_detections
                    if cat in TARGET_OBJECTS
                ]

            temporal_buffer.add_frame(raw_detections)
            merged_detections = temporal_buffer.get_merged_detections(raw_detections)
            last_raw_detections = raw_detections

            # Update tracker
            display_objects = tracker.update(merged_detections)

            # ========================================
            # 2. POSE ESTIMATION (Synchronous)
            # ========================================
            pose_inference_ms = 0
            pose_detected = False

            if pose_enabled:
                pose_frame_counter += 1
                skip_pose = (pose_frame_counter % POSE_INTERVAL) != 0

                pose_results = pose_detector.detect(frame, skip=skip_pose)
                pose_inference_ms = pose_detector.last_inference_time
                pose_detected = pose_detector.pose_detected

                # Draw pose SEBELUM bounding box agar bbox di atas pose
                pose_detector.draw_pose(frame, pose_results)

            # ========================================
            # 3. DRAW OBJECT TRACKING RESULTS
            # ========================================
            for obj in display_objects:
                if obj.category_name not in category_color_map:
                    category_color_map[obj.category_name] = get_color(color_index)
                    color_index += 1

                color = category_color_map[obj.category_name]
                draw_detection(frame, obj, color)

            # ========================================
            # 4. FPS CALCULATION
            # ========================================
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # ========================================
            # 5. HUD
            # ========================================
            draw_hud(
                frame, fps,
                len(last_raw_detections),
                len(display_objects),
                len(tracker.tracked_objects),
                pose_enabled,
                pose_detected,
                pose_inference_ms,
            )

            # Display
            cv2.imshow("Object Detection + Pose Estimation | Ultra-Stable v3", frame)

            # ========================================
            # 6. KEYBOARD INPUT
            # ========================================
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                break

            elif key == ord('p'):  # Toggle pose
                pose_enabled = not pose_enabled
                status = "ON" if pose_enabled else "OFF"
                print(f"[INFO] Pose Detection: {status}")

            elif key == ord('s'):  # Cycle pose style
                pose_style_idx = (pose_style_idx + 1) % len(pose_styles)
                POSE_DRAW_STYLE = pose_styles[pose_style_idx]
                print(f"[INFO] Pose Style: {POSE_DRAW_STYLE}")

    finally:
        # Cleanup
        detector.close()
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Program selesai.")


if __name__ == "__main__":
    main()
