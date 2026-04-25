"""MediaPipe Object Detection + Pose + Hand Tracking v4 (Compact)"""

import cv2, mediapipe as mp, numpy as np, time, sys, os
from collections import deque

# === CONFIG ===
MODEL_PATH, POSE_MODEL_PATH, HAND_MODEL_PATH = "efficientdet_lite0.tflite", "pose_landmarker_lite.task", "hand_landmarker.task"
MAX_RESULTS, SCORE_THRESHOLD, CAMERA_INDEX = 10, 0.20, 1
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
ENABLE_POSE, ENABLE_HAND = True, True
POSE_MIN_DETECTION_CONF, POSE_MIN_TRACKING_CONF = 0.5, 0.5
POSE_INTERVAL, POSE_SCALE_FACTOR, POSE_DRAW_STYLE = 1, 1.0, "full"
POSE_KALMAN_ENABLED = True
POSE_KALMAN_PROCESS_NOISE = 5e-3    # Lebih kecil = lebih smooth (tapi lebih lag)
POSE_KALMAN_MEASUREMENT_NOISE = 5e-2
POSE_LANDMARK_COLOR, POSE_CONNECTION_COLOR = (0, 255, 128), (230, 216, 173)
POSE_LANDMARK_SIZE, POSE_CONNECTION_THICKNESS = 4, 2
HAND_NUM_HANDS, HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF, HAND_MIN_PRESENCE_CONF = 2, 0.5, 0.5, 0.5
HAND_INTERVAL, HAND_DRAW_STYLE = 1, "full"
HAND_LANDMARK_COLOR, HAND_CONNECTION_COLOR = (255, 100, 255), (200, 150, 255)
HAND_LANDMARK_SIZE, HAND_CONNECTION_THICKNESS = 3, 2
MAX_CENTROID_DISTANCE, SIZE_WEIGHT = 200, 0.3
PROCESS_NOISE, MEASUREMENT_NOISE = 1e-2, 1e-1
MAX_MISSING_FRAMES, MIN_HIT_STREAK = 30, 2
DISPLAY_THRESHOLD, BBOX_SMOOTHING, SCORE_SMOOTHING = 0.18, 0.3, 0.3
TARGET_OBJECTS = []

POSE_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),
    (15,17),(15,19),(15,21),(12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),
    (23,24),(23,25),(25,27),(27,29),(27,31),(29,31),(24,26),(26,28),(28,30),(28,32),(30,32),
])

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
])

HAND_FINGER_GROUPS = {
    'thumb': [(0,1),(1,2),(2,3),(3,4)], 'index': [(0,5),(5,6),(6,7),(7,8)],
    'middle': [(0,9),(9,10),(10,11),(11,12)], 'ring': [(0,13),(13,14),(14,15),(15,16)],
    'pinky': [(0,17),(17,18),(18,19),(19,20)], 'palm': [(5,9),(9,13),(13,17)],
}
HAND_FINGER_COLORS = {
    'thumb': (0,200,255), 'index': (0,255,150), 'middle': (255,255,0),
    'ring': (255,150,0), 'pinky': (255,0,150), 'palm': (200,200,200),
}
HAND_FINGERTIP_INDICES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
COLORS = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),
          (255,0,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255)]

MODEL_URLS = {
    MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
    POSE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def download_model(path):
    if os.path.exists(path): return True
    url = MODEL_URLS.get(path)
    if not url: return False
    print(f"[INFO] Mengunduh model: {path}...")
    try:
        import urllib.request; urllib.request.urlretrieve(url, path); return True
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh: {e}"); return False

def lm_xy(lm, w, h): return (int(lm.x * w), int(lm.y * h))
def vis(lm): v = getattr(lm, 'visibility', 0.0); return v if v is not None else 0.0


# === LANDMARK KALMAN FILTER (untuk Pose Smoothing) ===
class LandmarkKalmanFilter:
    """Kalman Filter 4D untuk single landmark: state = [x, y, vx, vy]"""
    def __init__(self, x=0.0, y=0.0):
        self.state = np.array([x, y, 0, 0], dtype=np.float64)
        self.F = np.eye(4); self.F[0,2] = self.F[1,3] = 1  # position += velocity
        self.H = np.zeros((2,4)); self.H[0,0] = self.H[1,1] = 1
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * POSE_KALMAN_PROCESS_NOISE
        self.Q[2,2] = self.Q[3,3] = POSE_KALMAN_PROCESS_NOISE * 2  # velocity noise higher
        self.R = np.eye(2) * POSE_KALMAN_MEASUREMENT_NOISE
        self.initialized = False

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        if not self.initialized:
            self.state[:2] = [x, y]; self.initialized = True; return
        self.predict()
        z = np.array([x, y], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ (z - self.H @ self.state)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def x(self): return self.state[0]
    @property
    def y(self): return self.state[1]


class PoseSmoother:
    """Mengelola 33 LandmarkKalmanFilter untuk seluruh pose."""
    def __init__(self, num_landmarks=33):
        self.filters = [LandmarkKalmanFilter() for _ in range(num_landmarks)]
        self.num_landmarks = num_landmarks

    def smooth(self, landmarks):
        """Terima raw landmarks, return smoothed landmarks (as list of objects with .x, .y, .visibility)."""
        if landmarks is None: return None
        smoothed = []
        for i, lm in enumerate(landmarks):
            if i >= self.num_landmarks: break
            v = vis(lm)
            if v >= 0.5:
                self.filters[i].update(lm.x, lm.y)
            # Buat objek landmark baru dengan koordinat smoothed
            smoothed.append(_SmoothedLandmark(
                self.filters[i].x if self.filters[i].initialized else lm.x,
                self.filters[i].y if self.filters[i].initialized else lm.y,
                v
            ))
        return smoothed

    def reset(self):
        self.filters = [LandmarkKalmanFilter() for _ in range(self.num_landmarks)]


class _SmoothedLandmark:
    """Lightweight landmark object pengganti NormalizedLandmark."""
    __slots__ = ('x', 'y', 'visibility')
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


# === KALMAN FILTER ===
class BBoxKalmanFilter:
    def __init__(self, bbox):
        x, y, w, h = bbox
        self.state = np.array([x+w/2, y+h/2, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.F = np.eye(8); self.F[0,4] = self.F[1,5] = self.F[2,6] = self.F[3,7] = 1
        self.H = np.zeros((4,8)); np.fill_diagonal(self.H, 1)
        self.P = np.eye(8)*10; self.P[4,4]=self.P[5,5]=self.P[6,6]=self.P[7,7]=100
        self.Q = np.eye(8)*PROCESS_NOISE; self.Q[0,0]=self.Q[1,1]=PROCESS_NOISE*0.5; self.Q[4,4]=self.Q[5,5]=PROCESS_NOISE*2
        self.R = np.eye(4)*MEASUREMENT_NOISE

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.state[2:4] = np.maximum(self.state[2:4], 10)
        return self.get_bbox()

    def update(self, bbox):
        x, y, w, h = bbox
        z = np.array([x+w/2, y+h/2, w, h], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ (z - self.H @ self.state)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.state[2:4] = np.maximum(self.state[2:4], 10)
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.state[:4]; return [cx-w/2, cy-h/2, w, h]
    def get_centroid(self): return self.state[:2].copy()
    def get_velocity(self): return self.state[4:6].copy()
    def get_predicted_centroid(self, steps=1):
        return np.array([self.state[0]+self.state[4]*steps, self.state[1]+self.state[5]*steps])


# === TRACKED OBJECT ===
class TrackedObject:
    _next_id = 1
    def __init__(self, bbox, category_name, score):
        self.id = TrackedObject._next_id; TrackedObject._next_id += 1
        self.category_name, self.score, self.smooth_score, self.peak_score = category_name, score, score, score
        self.kf = BBoxKalmanFilter(bbox)
        self.raw_bbox = self.smooth_bbox = list(bbox)
        self.score_history = deque([score], maxlen=30)
        self.hit_streak, self.missing_frames, self.total_hits, self.age = 1, 0, 1, 1
        self.last_size = (bbox[2], bbox[3])

    def predict(self): return self.kf.predict()

    def update(self, bbox, score):
        self.raw_bbox = list(bbox); self.score = score
        kf_bbox = self.kf.update(bbox)
        self.smooth_bbox = [BBOX_SMOOTHING*k + (1-BBOX_SMOOTHING)*s for k, s in zip(kf_bbox, self.smooth_bbox)]
        self.smooth_score = SCORE_SMOOTHING*score + (1-SCORE_SMOOTHING)*self.smooth_score
        self.peak_score = max(self.peak_score, score)
        self.score_history.append(score)
        self.hit_streak += 1; self.missing_frames = 0; self.total_hits += 1; self.age += 1
        self.last_size = (bbox[2], bbox[3])

    def mark_missing(self):
        self.hit_streak = 0; self.missing_frames += 1; self.age += 1
        pb = self.kf.get_bbox()
        self.smooth_bbox = [0.2*p + 0.8*s for p, s in zip(pb, self.smooth_bbox)]
        self.smooth_score *= (0.92 if self.total_hits > 5 else 0.85)

    def should_display(self):
        return self.total_hits >= MIN_HIT_STREAK and self.smooth_score >= DISPLAY_THRESHOLD and self.missing_frames <= MAX_MISSING_FRAMES

    def is_expired(self):
        grace = MAX_MISSING_FRAMES * (2 if self.total_hits > 30 else 1.5 if self.total_hits > 10 else 1)
        return self.missing_frames > grace

    def get_display_bbox(self): return [int(round(v)) for v in self.smooth_bbox]
    def get_opacity(self): return 1.0 if self.missing_frames == 0 else max(0.3, 1.0 - self.missing_frames/MAX_MISSING_FRAMES*0.7)
    def get_centroid(self): return self.kf.get_centroid()
    def get_predicted_centroid(self, s=1): return self.kf.get_predicted_centroid(s)


# === OBJECT TRACKER ===
class ObjectTracker:
    def __init__(self): self.tracked_objects = []

    @staticmethod
    def compute_iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
        if ix1 >= ix2 or iy1 >= iy2: return 0.0
        inter = (ix2-ix1)*(iy2-iy1); union = b1[2]*b1[3]+b2[2]*b2[3]-inter
        return inter/union if union > 0 else 0.0

    @staticmethod
    def compute_cost(obj, bbox, cat):
        if obj.category_name != cat: return float('inf')
        dx, dy, dw, dh = bbox
        pred = obj.get_predicted_centroid(1)
        dist = np.sqrt((pred[0]-(dx+dw/2))**2 + (pred[1]-(dy+dh/2))**2)
        if dist > MAX_CENTROID_DISTANCE: return float('inf')
        tw, th = obj.last_size
        size_sim = (min(dw,tw)/max(dw,tw,1) + min(dh,th)/max(dh,th,1)) / 2
        return dist/MAX_CENTROID_DISTANCE + (1-size_sim)*SIZE_WEIGHT

    def update(self, detections):
        for o in self.tracked_objects: o.predict()
        if not detections and not self.tracked_objects: return []
        if not detections:
            for o in self.tracked_objects: o.mark_missing()
            self.tracked_objects = [o for o in self.tracked_objects if not o.is_expired()]
            return [o for o in self.tracked_objects if o.should_display()]
        if not self.tracked_objects:
            self.tracked_objects = [TrackedObject(b,c,s) for b,c,s in detections]
            return [o for o in self.tracked_objects if o.should_display()]

        costs = sorted([(self.compute_cost(self.tracked_objects[t], detections[d][0], detections[d][1]), t, d)
                         for t in range(len(self.tracked_objects)) for d in range(len(detections))
                         if self.compute_cost(self.tracked_objects[t], detections[d][0], detections[d][1]) < float('inf')])
        mt, md = set(), set()
        for _, t, d in costs:
            if t not in mt and d not in md:
                self.tracked_objects[t].update(detections[d][0], detections[d][2]); mt.add(t); md.add(d)
        for t in range(len(self.tracked_objects)):
            if t not in mt: self.tracked_objects[t].mark_missing()
        for d in range(len(detections)):
            if d not in md: self.tracked_objects.append(TrackedObject(detections[d][0], detections[d][1], detections[d][2]))
        self.tracked_objects = [o for o in self.tracked_objects if not o.is_expired()]
        return [o for o in self.tracked_objects if o.should_display()]


# === ASYNC OBJECT DETECTOR ===
class AsyncDetector:
    def __init__(self):
        self._latest_result = None; self._ts = 0
        opts = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            max_results=MAX_RESULTS, score_threshold=SCORE_THRESHOLD,
            result_callback=lambda r, o, t: setattr(self, '_latest_result', r))
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(opts)

    def detect_async(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._ts += 1
        try: self.detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts)
        except: pass
        if self._latest_result is None: return []
        return [([d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height],
                 d.categories[0].category_name, d.categories[0].score) for d in self._latest_result.detections]

    def close(self): self.detector.close()


# === POSE DETECTOR ===
class PoseDetector:
    def __init__(self):
        download_model(POSE_MODEL_PATH)
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONF,
            min_tracking_confidence=POSE_MIN_TRACKING_CONF, min_pose_presence_confidence=0.5)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        self.last_landmarks = None; self.pose_detected = False; self._ts = 0; self.last_inference_time = 0
        self.smoother = PoseSmoother(33) if POSE_KALMAN_ENABLED else None

    def detect(self, frame, skip=False):
        if skip and self.last_landmarks is not None: return self.last_landmarks
        self._ts += 1
        f = frame if POSE_SCALE_FACTOR >= 1.0 else cv2.resize(frame, None, fx=POSE_SCALE_FACTOR, fy=POSE_SCALE_FACTOR)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        try: result = self.landmarker.detect_for_video(img, self._ts)
        except: self.last_inference_time = (time.perf_counter()-t0)*1000; self.pose_detected = False; return None
        self.last_inference_time = (time.perf_counter()-t0)*1000
        if result.pose_landmarks:
            raw_lms = result.pose_landmarks[0]
            self.last_landmarks = self.smoother.smooth(raw_lms) if self.smoother else raw_lms
            self.pose_detected = True
        else:
            self.pose_detected = False
            if self.smoother and self.last_landmarks:
                # Predict-only saat pose hilang (tetap tampilkan posisi terakhir smoothed)
                for f in self.smoother.filters:
                    if f.initialized: f.predict()
        return self.last_landmarks

    def draw_pose(self, frame, lms=None):
        lms = lms or self.last_landmarks
        if not lms: return
        h, w = frame.shape[:2]
        style = POSE_DRAW_STYLE

        if style == "minimal":
            conns = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
            for s, e in conns:
                if s >= len(lms) or e >= len(lms) or vis(lms[s]) < 0.5 or vis(lms[e]) < 0.5: continue
                cv2.line(frame, lm_xy(lms[s],w,h), lm_xy(lms[e],w,h), POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS+1)
            for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                if i >= len(lms) or vis(lms[i]) < 0.5: continue
                pt = lm_xy(lms[i],w,h)
                cv2.circle(frame, pt, POSE_LANDMARK_SIZE+2, (50,50,50), -1)
                cv2.circle(frame, pt, POSE_LANDMARK_SIZE+1, POSE_LANDMARK_COLOR, -1)
        elif style == "skeleton_only":
            for s, e in POSE_CONNECTIONS:
                if s >= len(lms) or e >= len(lms) or vis(lms[s]) < 0.5 or vis(lms[e]) < 0.5: continue
                cv2.line(frame, lm_xy(lms[s],w,h), lm_xy(lms[e],w,h), POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)
        else:  # full
            for s, e in POSE_CONNECTIONS:
                if s >= len(lms) or e >= len(lms) or vis(lms[s]) < 0.5 or vis(lms[e]) < 0.5: continue
                sp, ep = lm_xy(lms[s],w,h), lm_xy(lms[e],w,h)
                cv2.line(frame, sp, ep, (100,100,100), POSE_CONNECTION_THICKNESS+2)
                cv2.line(frame, sp, ep, POSE_CONNECTION_COLOR, POSE_CONNECTION_THICKNESS)
            for idx, lm in enumerate(lms):
                if vis(lm) < 0.5: continue
                pt = lm_xy(lm,w,h)
                if idx in (0,11,12,23,24): sz, col = POSE_LANDMARK_SIZE+2, (0,255,200)
                elif idx in (15,16,19,20,27,28,31,32): sz, col = POSE_LANDMARK_SIZE+1, (255,200,0)
                else: sz, col = POSE_LANDMARK_SIZE, POSE_LANDMARK_COLOR
                cv2.circle(frame, pt, sz+3, (50,50,50), -1)
                cv2.circle(frame, pt, sz, col, -1)
                cv2.circle(frame, pt, sz, (255,255,255), 1)

    def close(self): self.landmarker.close()


# === HAND DETECTOR ===
class HandDetector:
    def __init__(self):
        download_model(HAND_MODEL_PATH)
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=HAND_NUM_HANDS,
            min_hand_detection_confidence=HAND_MIN_DETECTION_CONF,
            min_hand_presence_confidence=HAND_MIN_PRESENCE_CONF,
            min_tracking_confidence=HAND_MIN_TRACKING_CONF)
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self.last_landmarks = None; self.last_handedness = None
        self.hands_detected = 0; self._ts = 0; self.last_inference_time = 0

    def detect(self, frame, skip=False):
        if skip and self.last_landmarks is not None: return self.last_landmarks
        self._ts += 1
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        try: result = self.landmarker.detect_for_video(img, self._ts)
        except: self.last_inference_time = (time.perf_counter()-t0)*1000; self.hands_detected = 0; return None
        self.last_inference_time = (time.perf_counter()-t0)*1000
        if result.hand_landmarks:
            self.last_landmarks = result.hand_landmarks; self.last_handedness = result.handedness
            self.hands_detected = len(result.hand_landmarks)
        else: self.hands_detected = 0
        return self.last_landmarks

    def _finger_color(self, s, e):
        for finger, conns in HAND_FINGER_GROUPS.items():
            if (s, e) in conns: return HAND_FINGER_COLORS.get(finger, HAND_CONNECTION_COLOR)
        return HAND_CONNECTION_COLOR

    def _get_hand_label(self, idx):
        if self.last_handedness and idx < len(self.last_handedness):
            cats = self.last_handedness[idx]
            if cats: return cats[0].category_name
        return ""

    def draw_hands(self, frame, lms_list=None):
        lms_list = lms_list or self.last_landmarks
        if not lms_list: return
        h, w = frame.shape[:2]
        for hi, lms in enumerate(lms_list):
            label = self._get_hand_label(hi)
            if HAND_DRAW_STYLE == "skeleton_only":
                for s, e in HAND_CONNECTIONS:
                    if s < len(lms) and e < len(lms):
                        cv2.line(frame, lm_xy(lms[s],w,h), lm_xy(lms[e],w,h), self._finger_color(s,e), HAND_CONNECTION_THICKNESS)
            elif HAND_DRAW_STYLE == "minimal":
                for s, e in [(0,5),(5,8),(0,9),(9,12),(0,13),(13,16),(0,17),(17,20),(0,1),(1,4)]:
                    if s < len(lms) and e < len(lms):
                        cv2.line(frame, lm_xy(lms[s],w,h), lm_xy(lms[e],w,h), HAND_CONNECTION_COLOR, HAND_CONNECTION_THICKNESS)
                for i in [0,4,8,12,16,20]:
                    if i < len(lms):
                        pt = lm_xy(lms[i],w,h)
                        cv2.circle(frame, pt, HAND_LANDMARK_SIZE+2, (50,50,50), -1)
                        cv2.circle(frame, pt, HAND_LANDMARK_SIZE+1, (0,255,255), -1)
                if label and lms:
                    wpt = lm_xy(lms[0],w,h)
                    cv2.putText(frame, label, (wpt[0]-20,wpt[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            else:  # full
                for s, e in HAND_CONNECTIONS:
                    if s >= len(lms) or e >= len(lms): continue
                    sp, ep = lm_xy(lms[s],w,h), lm_xy(lms[e],w,h)
                    cv2.line(frame, sp, ep, (50,50,50), HAND_CONNECTION_THICKNESS+2)
                    cv2.line(frame, sp, ep, self._finger_color(s,e), HAND_CONNECTION_THICKNESS)
                for idx, lm in enumerate(lms):
                    pt = lm_xy(lm,w,h)
                    if idx == 0: sz, col = HAND_LANDMARK_SIZE+3, (255,255,255)
                    elif idx in HAND_FINGERTIP_INDICES: sz, col = HAND_LANDMARK_SIZE+2, (0,255,255)
                    elif idx in (1,5,9,13,17): sz, col = HAND_LANDMARK_SIZE+1, (200,200,255)
                    else: sz, col = HAND_LANDMARK_SIZE, HAND_LANDMARK_COLOR
                    cv2.circle(frame, pt, sz+3, (40,40,40), -1)
                    cv2.circle(frame, pt, sz, col, -1)
                    cv2.circle(frame, pt, sz, (255,255,255), 1)
                if label and lms:
                    wpt = lm_xy(lms[0],w,h)
                    lc = (0,255,150) if label == "Left" else (150,150,255)
                    cv2.putText(frame, label, (wpt[0]-20,wpt[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, (wpt[0]-20,wpt[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2, cv2.LINE_AA)

    def get_finger_states(self, lms):
        if not lms or len(lms) < 21: return None
        states = {'Thumb': lms[4].x < lms[3].x}
        for tip, pip, name in [(8,6,'Index'),(12,10,'Middle'),(16,14,'Ring'),(20,18,'Pinky')]:
            states[name] = lms[tip].y < lms[pip].y
        return states

    def close(self): self.landmarker.close()


# === TEMPORAL BUFFER ===
class TemporalDetectionBuffer:
    def __init__(self, size=3): self.buffer = deque(maxlen=size)
    def add_frame(self, dets): self.buffer.append(dets)
    def get_merged(self, current):
        if not self.buffer: return current
        merged, cur_bboxes = list(current), [d[0] for d in current]
        for frame_dets in self.buffer:
            for b, c, s in frame_dets:
                if not any(ObjectTracker.compute_iou(b, cb) > 0.3 for cb in cur_bboxes):
                    merged.append((b, c, s*0.7))
        return merged


# === DRAWING ===
def draw_detection(frame, obj, color):
    bbox = obj.get_display_bbox()
    x, y, w, h = bbox
    op = obj.get_opacity()
    fh, fw = frame.shape[:2]
    x, y = max(0, min(x, fw-1)), max(0, min(y, fh-1))
    w, h = max(10, min(w, fw-x)), max(10, min(h, fh-y))
    ac = tuple(int(c*op) for c in color)

    cv2.rectangle(frame, (x,y), (x+w,y+h), ac, 2)
    cl, ct = min(30, max(8, w//4), max(8, h//4)), max(3, int(4*op))
    for cx, cy, hx, hy, vx, vy in [(x,y,x+cl,y,x,y+cl),(x+w,y,x+w-cl,y,x+w,y+cl),
                                     (x,y+h,x+cl,y+h,x,y+h-cl),(x+w,y+h,x+w-cl,y+h,x+w,y+h-cl)]:
        cv2.line(frame, (cx,cy), (hx,hy), ac, ct); cv2.line(frame, (cx,cy), (vx,vy), ac, ct)

    label = f"#{obj.id} {obj.category_name}: {obj.smooth_score:.0%}"
    if obj.missing_frames > 0: label += f" [~{obj.missing_frames}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th_), _ = cv2.getTextSize(label, font, 0.6, 2)
    ly = max(y-10, th_+5)
    cv2.rectangle(frame, (x, ly-th_-5), (x+tw+10, ly+5), ac, -1)
    cv2.putText(frame, label, (x+5, ly), font, 0.6, (255,255,255), 2, cv2.LINE_AA)

    if obj.missing_frames == 0:
        vel = obj.kf.get_velocity()
        if np.linalg.norm(vel) > 2:
            c = obj.get_centroid(); e = c + vel*5
            cv2.arrowedLine(frame, tuple(c.astype(int)), tuple(e.astype(int)), (0,200,255), 2, tipLength=0.3)


def draw_hud(frame, fps, det_n, disp_n, total_n, pose_on, pose_det, pose_ms, hand_on, hands_det, hand_ms):
    h, w = frame.shape[:2]
    hud_h = 175 + (35 if pose_on else 0) + (35 if hand_on else 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (400, hud_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    y = 40
    fps_col = (0,255,0) if fps >= 20 else (0,255,255) if fps >= 10 else (0,0,255)

    lines = [
        (f"FPS: {fps:.1f}", 0.7, fps_col, 2),
        (f"Raw Detections: {det_n}", 0.55, (255,255,255), 1),
        (f"Displayed Objects: {disp_n}", 0.55, (180,255,180), 1),
        (f"Total Trackers: {total_n}", 0.5, (180,220,255), 1),
        (f"Filter: {', '.join(TARGET_OBJECTS[:3]) if TARGET_OBJECTS else 'Semua Objek'}", 0.5, (200,200,200), 1),
        ("Kalman Filter + Centroid Tracking", 0.45, (0,255,128), 1),
    ]
    for txt, sc, col, th in lines:
        cv2.putText(frame, txt, (20, y), font, sc, col, th, aa); y += int(sc * 40)

    if pose_on:
        ps, pc = ("DETECTED",(0,255,200)) if pose_det else ("No Person",(100,100,100))
        cv2.putText(frame, f"Pose: {ps} ({pose_ms:.1f}ms)", (20,y), font, 0.5, pc, 1, aa); y += 18
        cv2.putText(frame, f"Pose Style: {POSE_DRAW_STYLE} | Model: Lite", (20,y), font, 0.4, (150,180,255), 1, aa); y += 18
    else:
        cv2.putText(frame, "Pose: OFF ('p' untuk ON)", (20,y), font, 0.5, (100,100,100), 1, aa); y += 18

    if hand_on:
        hs, hc = (f"{hands_det} hand(s)",(255,150,255)) if hands_det > 0 else ("No Hands",(100,100,100))
        cv2.putText(frame, f"Hand: {hs} ({hand_ms:.1f}ms)", (20,y), font, 0.5, hc, 1, aa); y += 18
        cv2.putText(frame, f"Hand Style: {HAND_DRAW_STYLE} | Max: {HAND_NUM_HANDS}", (20,y), font, 0.4, (200,150,255), 1, aa)
    else:
        cv2.putText(frame, "Hand: OFF ('h' untuk ON)", (20,y), font, 0.5, (100,100,100), 1, aa)

    cv2.putText(frame, "'q'=Quit 'p'=Pose 'h'=Hand 's'=PoseStyle 'd'=HandStyle",
                (w-520, h-15), font, 0.45, (150,150,150), 1, aa)


# === MAIN ===
def main():
    global POSE_DRAW_STYLE, HAND_DRAW_STYLE
    print("=" * 60 + "\n  MediaPipe Object Detection + Pose + Hand Tracking v4\n" + "=" * 60)
    if not download_model(MODEL_PATH): sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    #cap = cv2.VideoCapture(r"C:\Users\Victus\Videos\Video Pokeb\Video Testing Dance.mp4")
    
    if not cap.isOpened(): print("[ERROR] Tidak dapat membuka kamera!"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[INFO] Resolusi: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("[INFO] 'q'=Quit 'p'=Pose 'h'=Hand 's'=PoseStyle 'd'=HandStyle\n")

    detector = AsyncDetector(); tracker = ObjectTracker(); buffer = TemporalDetectionBuffer()
    pose = PoseDetector(); hand = HandDetector()
    pose_on, hand_on = ENABLE_POSE, ENABLE_HAND
    styles_p, styles_h = ["full","minimal","skeleton_only"], ["full","minimal","skeleton_only"]
    si_p = styles_p.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in styles_p else 0
    si_h = styles_h.index(HAND_DRAW_STYLE) if HAND_DRAW_STYLE in styles_h else 0
    fps, fc, t0 = 0, 0, time.time()
    cat_colors, ci = {}, 0
    pfc, hfc = 0, 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # Object detection
            raw = detector.detect_async(frame)
            if TARGET_OBJECTS: raw = [(b,c,s) for b,c,s in raw if c in TARGET_OBJECTS]
            buffer.add_frame(raw)
            display_objs = tracker.update(buffer.get_merged(raw))

            # Pose
            pose_ms, pose_det = 0, False
            if pose_on:
                pfc += 1; pr = pose.detect(frame, skip=(pfc % POSE_INTERVAL != 0))
                pose_ms, pose_det = pose.last_inference_time, pose.pose_detected
                pose.draw_pose(frame, pr)

            # Hand
            hand_ms, hands_det = 0, 0
            if hand_on:
                hfc += 1; hr = hand.detect(frame, skip=(hfc % HAND_INTERVAL != 0))
                hand_ms, hands_det = hand.last_inference_time, hand.hands_detected
                hand.draw_hands(frame, hr)

            # Draw objects
            for obj in display_objs:
                if obj.category_name not in cat_colors: cat_colors[obj.category_name] = COLORS[ci % len(COLORS)]; ci += 1
                draw_detection(frame, obj, cat_colors[obj.category_name])

            # FPS
            fc += 1; elapsed = time.time() - t0
            if elapsed >= 1.0: fps = fc/elapsed; fc = 0; t0 = time.time()

            draw_hud(frame, fps, len(raw), len(display_objs), len(tracker.tracked_objects),
                     pose_on, pose_det, pose_ms, hand_on, hands_det, hand_ms)
            cv2.imshow("Object Detection + Pose + Hand | v4", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('p'): pose_on = not pose_on; print(f"[INFO] Pose: {'ON' if pose_on else 'OFF'}")
            elif key == ord('h'): hand_on = not hand_on; print(f"[INFO] Hand: {'ON' if hand_on else 'OFF'}")
            elif key == ord('s'): si_p = (si_p+1)%3; POSE_DRAW_STYLE = styles_p[si_p]; print(f"[INFO] Pose Style: {POSE_DRAW_STYLE}")
            elif key == ord('d'): si_h = (si_h+1)%3; HAND_DRAW_STYLE = styles_h[si_h]; print(f"[INFO] Hand Style: {HAND_DRAW_STYLE}")
    finally:
        detector.close(); pose.close(); hand.close(); cap.release(); cv2.destroyAllWindows()
        print("\n[INFO] Program selesai.")

if __name__ == "__main__":
    main()
