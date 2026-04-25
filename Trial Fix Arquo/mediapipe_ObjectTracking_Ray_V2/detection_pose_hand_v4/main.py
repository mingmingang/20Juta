"""
MediaPipe Object Detection + Pose Estimation + Hand Tracking v4
================================================================
Menggabungkan:
1. Object Detection (EfficientDet-Lite0) dengan Kalman Filter tracking
2. Pose Estimation (MediaPipe Pose) untuk deteksi skeleton tubuh
3. Hand Tracking (MediaPipe HandLandmarker) untuk deteksi tangan & jari

Model: EfficientDet-Lite0 + PoseLandmarker Lite + HandLandmarker
Framework: MediaPipe Tasks API + OpenCV
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
MAX_RESULTS = 10
SCORE_THRESHOLD = 0.20
CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ============================================================
# KONFIGURASI POSE ESTIMATION
# ============================================================
ENABLE_POSE = True
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5
POSE_INTERVAL = 1
POSE_SCALE_FACTOR = 1.0
POSE_DRAW_STYLE = "full"

POSE_LANDMARK_COLOR = (0, 255, 128)
POSE_CONNECTION_COLOR = (230, 216, 173)
POSE_LANDMARK_SIZE = 4
POSE_CONNECTION_THICKNESS = 2

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
# KONFIGURASI HAND TRACKING (BARU)
# ============================================================
ENABLE_HAND = True
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_NUM_HANDS = 2
HAND_MIN_DETECTION_CONF = 0.5
HAND_MIN_TRACKING_CONF = 0.5
HAND_MIN_PRESENCE_CONF = 0.5
HAND_INTERVAL = 1               # Jalankan hand detection setiap N frame
HAND_DRAW_STYLE = "full"   # "full", "minimal", "skeleton_only"

# Warna untuk hand tracking
HAND_LANDMARK_COLOR = (255, 100, 255)       # Pink/magenta untuk landmark
HAND_CONNECTION_COLOR = (200, 150, 255)      # Ungu muda untuk connections
HAND_LANDMARK_SIZE = 3
HAND_CONNECTION_THICKNESS = 2

# Warna per jari
HAND_THUMB_COLOR = (0, 200, 255)     # Orange
HAND_INDEX_COLOR = (0, 255, 150)     # Hijau terang
HAND_MIDDLE_COLOR = (255, 255, 0)    # Cyan
HAND_RING_COLOR = (255, 150, 0)      # Biru muda
HAND_PINKY_COLOR = (255, 0, 150)     # Magenta

# Hand connections (21 landmarks)
HAND_CONNECTIONS = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
])

# Mapping koneksi ke warna per jari
HAND_FINGER_GROUPS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],
    'palm': [(5, 9), (9, 13), (13, 17)],
}

HAND_FINGER_COLORS = {
    'thumb': HAND_THUMB_COLOR,
    'index': HAND_INDEX_COLOR,
    'middle': HAND_MIDDLE_COLOR,
    'ring': HAND_RING_COLOR,
    'pinky': HAND_PINKY_COLOR,
    'palm': (200, 200, 200),
}

# Finger tip indices (untuk label fingertip)
HAND_FINGERTIP_INDICES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

# ============================================================
# KONFIGURASI TRACKING & STABILISASI
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

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255),
]

TARGET_OBJECTS = []


# ============================================================
# KALMAN FILTER untuk Bounding Box Tracking
# ============================================================
class BBoxKalmanFilter:
    def __init__(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2.0, y + h / 2.0
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1
        self.H = np.zeros((4, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1
        self.P = np.eye(8) * 10.0
        self.P[4, 4] = self.P[5, 5] = self.P[6, 6] = self.P[7, 7] = 100.0
        self.Q = np.eye(8) * PROCESS_NOISE
        self.Q[0, 0] = self.Q[1, 1] = PROCESS_NOISE * 0.5
        self.Q[4, 4] = self.Q[5, 5] = PROCESS_NOISE * 2
        self.R = np.eye(4) * MEASUREMENT_NOISE

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.state[2] = max(self.state[2], 10)
        self.state[3] = max(self.state[3], 10)
        return self.get_bbox()

    def update(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - self.H @ self.state)
        self.P = (np.eye(8) - K @ self.H) @ self.P
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
        return np.array([self.state[0] + self.state[4] * steps,
                         self.state[1] + self.state[5] * steps])


# ============================================================
# TRACKED OBJECT CLASS
# ============================================================
class TrackedObject:
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
        return self.kf.predict()

    def update(self, bbox, score):
        self.raw_bbox = list(bbox)
        self.score = score
        kf_bbox = self.kf.update(bbox)
        for i in range(4):
            self.smooth_bbox[i] = BBOX_SMOOTHING * kf_bbox[i] + (1 - BBOX_SMOOTHING) * self.smooth_bbox[i]
        self.smooth_score = SCORE_SMOOTHING * score + (1 - SCORE_SMOOTHING) * self.smooth_score
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
        for i in range(4):
            self.smooth_bbox[i] = 0.2 * predicted_bbox[i] + 0.8 * self.smooth_bbox[i]
        self.smooth_score *= (0.92 if self.total_hits > 5 else 0.85)

    def should_display(self):
        return (self.total_hits >= MIN_HIT_STREAK and
                self.smooth_score >= DISPLAY_THRESHOLD and
                self.missing_frames <= MAX_MISSING_FRAMES)

    def is_expired(self):
        grace = MAX_MISSING_FRAMES
        if self.total_hits > 10: grace = int(grace * 1.5)
        if self.total_hits > 30: grace = MAX_MISSING_FRAMES * 2
        return self.missing_frames > grace

    def get_display_bbox(self):
        return [int(round(v)) for v in self.smooth_bbox]

    def get_opacity(self):
        if self.missing_frames == 0: return 1.0
        return max(0.3, 1.0 - (self.missing_frames / MAX_MISSING_FRAMES) * 0.7)

    def get_centroid(self):
        return self.kf.get_centroid()

    def get_predicted_centroid(self, steps=1):
        return self.kf.get_predicted_centroid(steps)


# ============================================================
# OBJECT TRACKER
# ============================================================
class ObjectTracker:
    def __init__(self):
        self.tracked_objects = []

    @staticmethod
    def compute_matching_cost(tracked_obj, det_bbox, det_category):
        if tracked_obj.category_name != det_category:
            return float('inf')
        dx, dy, dw, dh = det_bbox
        det_cx, det_cy = dx + dw / 2.0, dy + dh / 2.0
        pred_cx, pred_cy = tracked_obj.get_predicted_centroid(steps=1)
        dist = np.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2)
        if dist > MAX_CENTROID_DISTANCE:
            return float('inf')
        tw, th = tracked_obj.last_size
        size_sim = (min(dw, tw) / max(dw, tw, 1) + min(dh, th) / max(dh, th, 1)) / 2.0
        return dist / MAX_CENTROID_DISTANCE + (1.0 - size_sim) * SIZE_WEIGHT

    @staticmethod
    def compute_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        ix1, iy1 = max(x1, x2), max(y1, y2)
        ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if ix1 >= ix2 or iy1 >= iy2: return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        for obj in self.tracked_objects:
            obj.predict()

        if not detections and not self.tracked_objects:
            return []
        if not detections:
            for obj in self.tracked_objects:
                obj.mark_missing()
            self.tracked_objects = [o for o in self.tracked_objects if not o.is_expired()]
            return [o for o in self.tracked_objects if o.should_display()]
        if not self.tracked_objects:
            for bbox, cat, score in detections:
                self.tracked_objects.append(TrackedObject(bbox, cat, score))
            return [o for o in self.tracked_objects if o.should_display()]

        nt, nd = len(self.tracked_objects), len(detections)
        costs = []
        for t in range(nt):
            for d in range(nd):
                c = self.compute_matching_cost(self.tracked_objects[t], detections[d][0], detections[d][1])
                if c < float('inf'):
                    costs.append((c, t, d))
        costs.sort(key=lambda x: x[0])

        mt, md, matches = set(), set(), []
        for cost, t, d in costs:
            if t not in mt and d not in md:
                matches.append((t, d)); mt.add(t); md.add(d)

        for t, d in matches:
            self.tracked_objects[t].update(detections[d][0], detections[d][2])
        for t in range(nt):
            if t not in mt: self.tracked_objects[t].mark_missing()
        for d in range(nd):
            if d not in md:
                self.tracked_objects.append(TrackedObject(detections[d][0], detections[d][1], detections[d][2]))

        self.tracked_objects = [o for o in self.tracked_objects if not o.is_expired()]
        return [o for o in self.tracked_objects if o.should_display()]


# ============================================================
# ASYNC DETECTOR (LIVE_STREAM mode)
# ============================================================
class AsyncDetector:
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
            for det in self._latest_result.detections:
                detections.append(([det.bounding_box.origin_x, det.bounding_box.origin_y,
                                    det.bounding_box.width, det.bounding_box.height],
                                   det.categories[0].category_name,
                                   det.categories[0].score))
        return detections

    def close(self):
        self.detector.close()


# ============================================================
# POSE DETECTOR
# ============================================================
class PoseDetector:
    def __init__(self, model_path, min_detection_conf=0.5, min_tracking_conf=0.5):
        self._download_model(model_path,
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task")

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            min_pose_presence_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_landmarks = None
        self.pose_detected = False
        self._timestamp = 0
        self.last_inference_time = 0

    def _download_model(self, path, url):
        if os.path.exists(path): return
        print(f"[INFO] Mengunduh model: {path}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"[INFO] Model berhasil diunduh: {path}")
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh: {e}")

    def detect(self, frame, skip=False):
        if skip and self.last_landmarks is not None:
            return self.last_landmarks
        self._timestamp += 1

        if POSE_SCALE_FACTOR < 1.0:
            h, w = frame.shape[:2]
            pose_frame = cv2.resize(frame, (int(w * POSE_SCALE_FACTOR), int(h * POSE_SCALE_FACTOR)))
        else:
            pose_frame = frame

        rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        try:
            result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        except Exception:
            self.last_inference_time = (time.perf_counter() - t0) * 1000
            self.pose_detected = False
            return None
        self.last_inference_time = (time.perf_counter() - t0) * 1000

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.last_landmarks = result.pose_landmarks[0]
            self.pose_detected = True
        else:
            self.pose_detected = False
        return self.last_landmarks

    def _lm_xy(self, lm, w, h):
        return (int(lm.x * w), int(lm.y * h))

    def _vis(self, lm):
        v = getattr(lm, 'visibility', 0.0)
        return v if v is not None else 0.0

    def draw_pose(self, frame, landmarks=None):
        lms = landmarks or self.last_landmarks
        if lms is None: return
        h, w = frame.shape[:2]

        if POSE_DRAW_STYLE == "minimal":
            conns = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
            keys = [11,12,13,14,15,16,23,24,25,26,27,28]
            for s, e in conns:
                if s >= len(lms) or e >= len(lms): continue
                if self._vis(lms[s]) < 0.5 or self._vis(lms[e]) < 0.5: continue
                cv2.line(frame, self._lm_xy(lms[s], w, h), self._lm_xy(lms[e], w, h),
                         POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS + 1)
            for i in keys:
                if i >= len(lms) or self._vis(lms[i]) < 0.5: continue
                pt = self._lm_xy(lms[i], w, h)
                cv2.circle(frame, pt, POSE_LANDMARK_SIZE + 2, (50,50,50), -1)
                cv2.circle(frame, pt, POSE_LANDMARK_SIZE + 1, POSE_LANDMARK_COLOR, -1)
        elif POSE_DRAW_STYLE == "skeleton_only":
            for s, e in POSE_CONNECTIONS:
                if s >= len(lms) or e >= len(lms): continue
                if self._vis(lms[s]) < 0.5 or self._vis(lms[e]) < 0.5: continue
                cv2.line(frame, self._lm_xy(lms[s], w, h), self._lm_xy(lms[e], w, h),
                         POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)
        else:  # full
            for s, e in POSE_CONNECTIONS:
                if s >= len(lms) or e >= len(lms): continue
                if self._vis(lms[s]) < 0.5 or self._vis(lms[e]) < 0.5: continue
                sp, ep = self._lm_xy(lms[s], w, h), self._lm_xy(lms[e], w, h)
                cv2.line(frame, sp, ep, (100,100,100), POSE_CONNECTION_THICKNESS + 2)
                cv2.line(frame, sp, ep, POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)
            for idx, lm in enumerate(lms):
                if self._vis(lm) < 0.5: continue
                pt = self._lm_xy(lm, w, h)
                if idx in [0, 11, 12, 23, 24]:
                    sz, col = POSE_LANDMARK_SIZE + 2, (0, 255, 200)
                elif idx in [15, 16, 19, 20, 27, 28, 31, 32]:
                    sz, col = POSE_LANDMARK_SIZE + 1, (255, 200, 0)
                else:
                    sz, col = POSE_LANDMARK_SIZE, POSE_LANDMARK_COLOR
                cv2.circle(frame, pt, sz + 3, (50, 50, 50), -1)
                cv2.circle(frame, pt, sz, col, -1)
                cv2.circle(frame, pt, sz, (255, 255, 255), 1)

    def close(self):
        self.landmarker.close()


# ============================================================
# HAND DETECTOR (BARU - MediaPipe Tasks API HandLandmarker)
# ============================================================
class HandDetector:
    """
    Wrapper untuk MediaPipe Tasks API HandLandmarker.
    Deteksi tangan dengan 21 landmark per tangan.
    Mendukung hingga 2 tangan sekaligus.
    """

    def __init__(self, model_path, num_hands=2,
                 min_detection_conf=0.5, min_tracking_conf=0.5,
                 min_presence_conf=0.5):
        self._download_model(model_path)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.last_landmarks = None   # list of list of NormalizedLandmark
        self.last_handedness = None  # list of handedness (Left/Right)
        self.hands_detected = 0
        self._timestamp = 0
        self.last_inference_time = 0

    def _download_model(self, path):
        if os.path.exists(path): return
        print(f"[INFO] Mengunduh model hand: {path}...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/"
               "hand_landmarker.task")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"[INFO] Model hand berhasil diunduh: {path}")
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh model hand: {e}")

    def detect(self, frame, skip=False):
        if skip and self.last_landmarks is not None:
            return self.last_landmarks
        self._timestamp += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        try:
            result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        except Exception:
            self.last_inference_time = (time.perf_counter() - t0) * 1000
            self.hands_detected = 0
            return None
        self.last_inference_time = (time.perf_counter() - t0) * 1000

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            self.last_landmarks = result.hand_landmarks
            self.last_handedness = result.handedness
            self.hands_detected = len(result.hand_landmarks)
        else:
            self.hands_detected = 0

        return self.last_landmarks

    def _lm_xy(self, lm, w, h):
        return (int(lm.x * w), int(lm.y * h))

    def _get_finger_color(self, start_idx, end_idx):
        """Mendapatkan warna berdasarkan jari."""
        for finger, conns in HAND_FINGER_GROUPS.items():
            if (start_idx, end_idx) in conns:
                return HAND_FINGER_COLORS.get(finger, HAND_CONNECTION_COLOR)
        return HAND_CONNECTION_COLOR

    def draw_hands(self, frame, landmarks_list=None):
        lms_list = landmarks_list or self.last_landmarks
        if lms_list is None: return

        h, w = frame.shape[:2]

        for hand_idx, lms in enumerate(lms_list):
            # Tentukan label handedness
            hand_label = ""
            if self.last_handedness and hand_idx < len(self.last_handedness):
                cats = self.last_handedness[hand_idx]
                if cats and len(cats) > 0:
                    hand_label = cats[0].category_name  # "Left" or "Right"

            if HAND_DRAW_STYLE == "full":
                self._draw_full_hand(frame, lms, w, h, hand_label, hand_idx)
            elif HAND_DRAW_STYLE == "minimal":
                self._draw_minimal_hand(frame, lms, w, h, hand_label, hand_idx)
            elif HAND_DRAW_STYLE == "skeleton_only":
                self._draw_skeleton_hand(frame, lms, w, h)

    def _draw_full_hand(self, frame, lms, w, h, hand_label, hand_idx):
        """Draw semua landmark dan koneksi dengan warna per jari."""
        # Draw connections dengan warna per jari
        for s, e in HAND_CONNECTIONS:
            if s >= len(lms) or e >= len(lms): continue
            sp = self._lm_xy(lms[s], w, h)
            ep = self._lm_xy(lms[e], w, h)
            color = self._get_finger_color(s, e)
            cv2.line(frame, sp, ep, (50, 50, 50), HAND_CONNECTION_THICKNESS + 2)
            cv2.line(frame, sp, ep, color, HAND_CONNECTION_THICKNESS)

        # Draw landmarks
        for idx, lm in enumerate(lms):
            pt = self._lm_xy(lm, w, h)

            if idx == 0:  # Wrist
                sz = HAND_LANDMARK_SIZE + 3
                col = (255, 255, 255)
            elif idx in HAND_FINGERTIP_INDICES:  # Fingertips
                sz = HAND_LANDMARK_SIZE + 2
                col = (0, 255, 255)
            elif idx in [1, 5, 9, 13, 17]:  # Knuckles (base)
                sz = HAND_LANDMARK_SIZE + 1
                col = (200, 200, 255)
            else:
                sz = HAND_LANDMARK_SIZE
                col = HAND_LANDMARK_COLOR

            # Glow effect
            cv2.circle(frame, pt, sz + 3, (40, 40, 40), -1)
            cv2.circle(frame, pt, sz, col, -1)
            cv2.circle(frame, pt, sz, (255, 255, 255), 1)

        # Draw hand label (Left/Right)
        if hand_label and 0 < len(lms):
            wrist_pt = self._lm_xy(lms[0], w, h)
            # Label warna berbeda per tangan
            label_color = (0, 255, 150) if hand_label == "Left" else (150, 150, 255)
            cv2.putText(frame, hand_label, (wrist_pt[0] - 20, wrist_pt[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hand_label, (wrist_pt[0] - 20, wrist_pt[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)

    def _draw_minimal_hand(self, frame, lms, w, h, hand_label, hand_idx):
        """Draw hanya fingertips dan koneksi utama."""
        key_conns = [(0,5),(5,8),(0,9),(9,12),(0,13),(13,16),(0,17),(17,20),(0,1),(1,4)]
        for s, e in key_conns:
            if s >= len(lms) or e >= len(lms): continue
            cv2.line(frame, self._lm_xy(lms[s], w, h), self._lm_xy(lms[e], w, h),
                     HAND_CONNECTION_COLOR, HAND_CONNECTION_THICKNESS)

        for i in [0, 4, 8, 12, 16, 20]:
            if i >= len(lms): continue
            pt = self._lm_xy(lms[i], w, h)
            cv2.circle(frame, pt, HAND_LANDMARK_SIZE + 2, (50,50,50), -1)
            cv2.circle(frame, pt, HAND_LANDMARK_SIZE + 1, (0, 255, 255), -1)

        if hand_label and len(lms) > 0:
            wpt = self._lm_xy(lms[0], w, h)
            cv2.putText(frame, hand_label, (wpt[0]-20, wpt[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    def _draw_skeleton_hand(self, frame, lms, w, h):
        """Draw hanya garis koneksi."""
        for s, e in HAND_CONNECTIONS:
            if s >= len(lms) or e >= len(lms): continue
            color = self._get_finger_color(s, e)
            cv2.line(frame, self._lm_xy(lms[s], w, h), self._lm_xy(lms[e], w, h),
                     color, HAND_CONNECTION_THICKNESS)

    def get_finger_states(self, hand_landmarks):
        """
        Deteksi jari mana yang terbuka/tertutup.
        Returns dict: {'Thumb': True/False, 'Index': True/False, ...}
        """
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return None

        states = {}
        # Thumb: compare x of tip (4) vs IP joint (3)
        # (simplified - works for right hand facing camera)
        states['Thumb'] = hand_landmarks[4].x < hand_landmarks[3].x

        # Other fingers: compare y of tip vs PIP joint
        finger_tips = [(8, 6, 'Index'), (12, 10, 'Middle'),
                       (16, 14, 'Ring'), (20, 18, 'Pinky')]
        for tip, pip, name in finger_tips:
            states[name] = hand_landmarks[tip].y < hand_landmarks[pip].y

        return states

    def close(self):
        self.landmarker.close()


# ============================================================
# TEMPORAL BUFFER
# ============================================================
class TemporalDetectionBuffer:
    def __init__(self, buffer_size=3):
        self.buffer = deque(maxlen=buffer_size)

    def add_frame(self, detections):
        self.buffer.append(detections)

    def get_merged_detections(self, current_detections):
        if not self.buffer: return current_detections
        merged = list(current_detections)
        cur_bboxes = [d[0] for d in current_detections]
        for frame_dets in self.buffer:
            for bbox, cat, score in frame_dets:
                if not any(ObjectTracker.compute_iou(bbox, cb) > 0.3 for cb in cur_bboxes):
                    merged.append((bbox, cat, score * 0.7))
        return merged


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Model sudah ada: {MODEL_PATH}")
        return True
    print("[INFO] Mengunduh model EfficientDet-Lite0...")
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "object_detector/efficientdet_lite0/int8/latest/"
           "efficientdet_lite0.tflite")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"[INFO] Model berhasil diunduh: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh model: {e}")
        return False


def get_color(index):
    return COLORS[index % len(COLORS)]


def draw_detection(frame, tracked_obj, color):
    bbox = tracked_obj.get_display_bbox()
    x, y, w, h = bbox
    opacity = tracked_obj.get_opacity()
    fh, fw = frame.shape[:2]
    x, y = max(0, min(x, fw-1)), max(0, min(y, fh-1))
    w, h = max(10, min(w, fw-x)), max(10, min(h, fh-y))

    adj_color = tuple(int(c * opacity) for c in color)
    cv2.rectangle(frame, (x, y), (x+w, y+h), adj_color, 2)

    cl = min(30, max(8, w//4), max(8, h//4))
    ct = max(3, int(4 * opacity))
    for corner, h_end, v_end in [
        ((x,y),(x+cl,y),(x,y+cl)), ((x+w,y),(x+w-cl,y),(x+w,y+cl)),
        ((x,y+h),(x+cl,y+h),(x,y+h-cl)), ((x+w,y+h),(x+w-cl,y+h),(x+w,y+h-cl))]:
        cv2.line(frame, corner, h_end, adj_color, ct)
        cv2.line(frame, corner, v_end, adj_color, ct)

    label = f"#{tracked_obj.id} {tracked_obj.category_name}: {tracked_obj.smooth_score:.0%}"
    if tracked_obj.missing_frames > 0:
        label += f" [~{tracked_obj.missing_frames}]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
    ly = max(y - 10, th + 5)
    cv2.rectangle(frame, (x, ly-th-5), (x+tw+10, ly+5), adj_color, -1)
    cv2.putText(frame, label, (x+5, ly), font, 0.6, (255,255,255), 2, cv2.LINE_AA)

    if tracked_obj.missing_frames == 0:
        vel = tracked_obj.kf.get_velocity()
        if np.sqrt(vel[0]**2 + vel[1]**2) > 2:
            c = tracked_obj.get_centroid()
            e = c + vel * 5
            cv2.arrowedLine(frame, (int(c[0]),int(c[1])), (int(e[0]),int(e[1])),
                            (0,200,255), 2, tipLength=0.3)


def draw_hud(frame, fps, det_count, disp_count, total_trackers,
             pose_enabled, pose_detected, pose_ms,
             hand_enabled, hands_detected, hand_ms):
    """Draw HUD with tracking + pose + hand info."""
    h, w = frame.shape[:2]

    # Hitung tinggi HUD
    hud_h = 175
    if pose_enabled: hud_h += 35
    if hand_enabled: hud_h += 35

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_pos = 40
    fps_col = (0,255,0) if fps >= 20 else (0,255,255) if fps >= 10 else (0,0,255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_col, 2, cv2.LINE_AA)

    y_pos += 30
    cv2.putText(frame, f"Raw Detections: {det_count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(frame, f"Displayed Objects: {disp_count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 1, cv2.LINE_AA)
    y_pos += 20
    cv2.putText(frame, f"Total Trackers: {total_trackers}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,220,255), 1, cv2.LINE_AA)

    y_pos += 22
    if TARGET_OBJECTS:
        ft = f"Filter: {', '.join(TARGET_OBJECTS[:3])}"
        if len(TARGET_OBJECTS) > 3: ft += f" +{len(TARGET_OBJECTS)-3}"
    else:
        ft = "Filter: Semua Objek"
    cv2.putText(frame, ft, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    y_pos += 20
    cv2.putText(frame, "Kalman Filter + Centroid Tracking", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,128), 1, cv2.LINE_AA)

    # Pose info
    y_pos += 22
    if pose_enabled:
        ps = "DETECTED" if pose_detected else "No Person"
        pc = (0,255,200) if pose_detected else (100,100,100)
        cv2.putText(frame, f"Pose: {ps} ({pose_ms:.1f}ms)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pc, 1, cv2.LINE_AA)
        y_pos += 18
        cv2.putText(frame, f"Pose Style: {POSE_DRAW_STYLE} | Model: Lite", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,180,255), 1, cv2.LINE_AA)
        y_pos += 18
    else:
        cv2.putText(frame, "Pose: OFF ('p' untuk ON)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1, cv2.LINE_AA)
        y_pos += 18

    # Hand info
    if hand_enabled:
        hs = f"{hands_detected} hand(s)" if hands_detected > 0 else "No Hands"
        hc = (255, 150, 255) if hands_detected > 0 else (100, 100, 100)
        cv2.putText(frame, f"Hand: {hs} ({hand_ms:.1f}ms)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hc, 1, cv2.LINE_AA)
        y_pos += 18
        cv2.putText(frame, f"Hand Style: {HAND_DRAW_STYLE} | Max: {HAND_NUM_HANDS}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,150,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Hand: OFF ('h' untuk ON)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "'q'=Quit 'p'=Pose 'h'=Hand 's'=PoseStyle 'd'=HandStyle",
                (w - 520, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================
def main():
    global POSE_DRAW_STYLE, HAND_DRAW_STYLE

    print("=" * 65)
    print("  MediaPipe Object Detection + Pose + Hand Tracking v4")
    print("=" * 65)

    if not download_model():
        sys.exit(1)

    print(f"\n[INFO] Membuka kamera (index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    #cap = cv2.VideoCapture(r"C:\Users\Victus\Videos\Video Pokeb\Video Testing Dance.mp4")
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolusi kamera: {actual_w}x{actual_h}")

    print(f"\n[INFO] Konfigurasi:")
    print(f"  === Object Detection ===")
    print(f"  - Mode: LIVE_STREAM (Async) + Kalman Filter")
    print(f"  === Pose Estimation ===")
    print(f"  - Enabled: {ENABLE_POSE} | Style: {POSE_DRAW_STYLE}")
    print(f"  === Hand Tracking ===")
    print(f"  - Enabled: {ENABLE_HAND} | Max Hands: {HAND_NUM_HANDS}")
    print(f"  - Style: {HAND_DRAW_STYLE}")
    print(f"\n[INFO] Kontrol keyboard:")
    print(f"  'q'/ESC = Keluar")
    print(f"  'p'     = Toggle Pose ON/OFF")
    print(f"  'h'     = Toggle Hand ON/OFF")
    print(f"  's'     = Ganti Pose Style")
    print(f"  'd'     = Ganti Hand Style")
    print(f"\n[INFO] Deteksi dimulai!\n")

    # Initialize
    detector = AsyncDetector(MODEL_PATH, MAX_RESULTS, SCORE_THRESHOLD)
    tracker_obj = ObjectTracker()
    temporal_buffer = TemporalDetectionBuffer(buffer_size=3)

    pose_detector = PoseDetector(POSE_MODEL_PATH, POSE_MIN_DETECTION_CONF, POSE_MIN_TRACKING_CONF)
    hand_detector = HandDetector(HAND_MODEL_PATH, HAND_NUM_HANDS,
                                 HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF,
                                 HAND_MIN_PRESENCE_CONF)

    pose_enabled = ENABLE_POSE
    hand_enabled = ENABLE_HAND
    pose_styles = ["full", "minimal", "skeleton_only"]
    hand_styles = ["full", "minimal", "skeleton_only"]
    pose_si = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0
    hand_si = hand_styles.index(HAND_DRAW_STYLE) if HAND_DRAW_STYLE in hand_styles else 0

    fps = 0
    frame_count = 0
    start_time = time.time()
    category_color_map = {}
    color_index = 0
    last_raw = []
    pose_fc = 0
    hand_fc = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)

            # 1. Object Detection
            raw = detector.detect_async(frame)
            if TARGET_OBJECTS:
                raw = [(b,c,s) for b,c,s in raw if c in TARGET_OBJECTS]
            temporal_buffer.add_frame(raw)
            merged = temporal_buffer.get_merged_detections(raw)
            last_raw = raw
            display_objs = tracker_obj.update(merged)

            # 2. Pose Estimation
            pose_ms, pose_det = 0, False
            if pose_enabled:
                pose_fc += 1
                pose_results = pose_detector.detect(frame, skip=(pose_fc % POSE_INTERVAL != 0))
                pose_ms = pose_detector.last_inference_time
                pose_det = pose_detector.pose_detected
                pose_detector.draw_pose(frame, pose_results)

            # 3. Hand Tracking
            hand_ms, hands_det = 0, 0
            if hand_enabled:
                hand_fc += 1
                hand_results = hand_detector.detect(frame, skip=(hand_fc % HAND_INTERVAL != 0))
                hand_ms = hand_detector.last_inference_time
                hands_det = hand_detector.hands_detected
                hand_detector.draw_hands(frame, hand_results)

            # 4. Draw object tracking
            for obj in display_objs:
                if obj.category_name not in category_color_map:
                    category_color_map[obj.category_name] = get_color(color_index)
                    color_index += 1
                draw_detection(frame, obj, category_color_map[obj.category_name])

            # 5. FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # 6. HUD
            draw_hud(frame, fps, len(last_raw), len(display_objs),
                     len(tracker_obj.tracked_objects),
                     pose_enabled, pose_det, pose_ms,
                     hand_enabled, hands_det, hand_ms)

            cv2.imshow("Object Detection + Pose + Hand | v4", frame)

            # 7. Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord('p'):
                pose_enabled = not pose_enabled
                print(f"[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")
            elif key == ord('h'):
                hand_enabled = not hand_enabled
                print(f"[INFO] Hand: {'ON' if hand_enabled else 'OFF'}")
            elif key == ord('s'):
                pose_si = (pose_si + 1) % len(pose_styles)
                POSE_DRAW_STYLE = pose_styles[pose_si]
                print(f"[INFO] Pose Style: {POSE_DRAW_STYLE}")
            elif key == ord('d'):
                hand_si = (hand_si + 1) % len(hand_styles)
                HAND_DRAW_STYLE = hand_styles[hand_si]
                print(f"[INFO] Hand Style: {HAND_DRAW_STYLE}")

    finally:
        detector.close()
        pose_detector.close()
        hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Program selesai.")


if __name__ == "__main__":
    main()
