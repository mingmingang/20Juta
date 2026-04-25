"""
V2.2 Person Tracking with ArUco Identification - T-MIND Project
================================================================
Menggabungkan:
1. YOLOv8 → deteksi semua orang (bounding box per person)
2. ArUco  → identifikasi operator (marker ID = identitas)
3. MediaPipe Holistic → skeleton pose HANYA di operator yang teridentifikasi

Alur:
  YOLO detect persons → ArUco detect marker → Match marker ke bbox →
  Crop bbox operator → MediaPipe Pose di crop → Remap skeleton ke frame asli

Kontrol:
  'q' / ESC  = Keluar
  'p'        = Toggle Pose ON/OFF (semua)
  's'        = Toggle Skeleton style (full/minimal)
  '1'        = Toggle Body Pose ON/OFF
  '2'        = Toggle Hands (jari) ON/OFF
  '3'        = Toggle Face Mesh (mata) ON/OFF
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import time
import sys
import threading

# ============================================================
# CONFIGURATION
# ============================================================

# --- Kamera ---
CAMERA_INDEX    = 0                     # Index kamera (0 = default, 1 = external)
FRAME_WIDTH     = 1280                   # Lebar frame (diturunkan untuk FPS)
FRAME_HEIGHT    = 720                   # Tinggi frame (diturunkan untuk FPS)

# --- YOLO Person Detection ---
YOLO_MODEL      = "yolov8n.pt"          # Model YOLO (auto-download jika belum ada)
YOLO_CONF       = 0.40                  # Confidence threshold person detection
YOLO_PERSON_CLASS = 0                   # Class ID untuk "person" di COCO

# --- ArUco Detection ---
ARUCO_DICT_TYPE = "4x4_50"              # Harus cocok dengan generator!

# --- Mapping ID → Nama Operator ---
ID_MAP = {
    0: "Arya Dwi Kusuma",
    1: "Person B",
    2: "Person C",
    3: "Person D",
    4: "Person E",
    # Tambah sesuai kebutuhan...
}
DEFAULT_NAME = "Unknown"

# --- Target Operator ---
# Set ke ID tertentu untuk HANYA track operator ini
# Set ke None untuk track siapa saja yang pegang ArUco
TARGET_OPERATOR_ID = None               # None = semua, 0 = hanya ID 0, dst.

# --- MediaPipe Pose ---
ENABLE_POSE         = True
ENABLE_BODY_POSE    = True              # Toggle body skeleton
ENABLE_HANDS        = True              # Toggle hand/finger landmarks
ENABLE_FACE         = True              # Toggle face mesh (mata, dll)
POSE_DRAW_STYLE     = "full"            # "full" atau "minimal"
CROP_PADDING        = 30                # Padding di sekeliling crop (piksel)

# --- Warna ---
YOLO_BBOX_COLOR     = (255, 180, 0)     # Warna bbox YOLO (non-operator) — biru muda
OPERATOR_BBOX_COLOR = (0, 255, 100)     # Warna bbox operator teridentifikasi — hijau
ARUCO_COLOR         = (0, 255, 255)     # Warna ArUco border — kuning
NAME_COLOR          = (0, 255, 255)     # Warna nama operator — kuning
SKELETON_COLOR_LM   = (245, 117, 66)    # Warna landmark pose
SKELETON_COLOR_CN   = (245, 66, 230)    # Warna connection pose

# ============================================================

# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "7x7_50":  cv2.aruco.DICT_7X7_50,
}


# ============================================================
# MULTITHREADING CAMERA
# ============================================================
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # buffer size 1 for minimum latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Resolusi Input Kamera: {self.actual_w}x{self.actual_h}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Jalankan loop baca kamera di thread terpisah agar tidak nunggu YOLO
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                return
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            self.ret = ret
            self.frame = frame

    def read(self):
        # Return copy of frame untuk mencegah error tertimpa oleh thread
        if self.ret and self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)  # Tunggu thread selesai
        self.cap.release()


# ============================================================
# ARUCO DETECTOR (dari V2.1)
# ============================================================
class ArUcoDetector:
    def __init__(self, dict_type=ARUCO_DICT_TYPE):
        dict_key = ARUCO_DICT_MAP.get(dict_type, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detect(self, frame):
        """Detect ArUco markers. Returns list of {id, name, corners, center}."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        results = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                name = ID_MAP.get(int(marker_id), DEFAULT_NAME)
                results.append({
                    'id': int(marker_id),
                    'name': name,
                    'corners': pts,
                    'center': (cx, cy),
                })
        return results


# ============================================================
# YOLO PERSON DETECTOR
# ============================================================
class YOLOPersonDetector:
    def __init__(self, model_path=YOLO_MODEL, conf=YOLO_CONF):
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_persons(self, frame):
        """
        Detect all persons in frame.
        Returns list of {bbox: (x1,y1,x2,y2), conf: float}
        """
        results = self.model(frame, conf=self.conf, classes=[YOLO_PERSON_CLASS],
                             verbose=False)
        persons = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                    })
        return persons


# ============================================================
# MATCHING & TRACKING: ArUco → YOLO Person
# ============================================================
def match_aruco_to_person(aruco_results, person_results):
    """
    Match setiap ArUco marker ke YOLO person bbox.
    ArUco center harus berada di dalam bbox person.

    Returns:
        list of dict: [{
            'person': person_dict,
            'aruco': aruco_dict,
        }, ...]
    """
    matches = []

    for ar in aruco_results:
        # Filter berdasarkan target operator
        if TARGET_OPERATOR_ID is not None and ar['id'] != TARGET_OPERATOR_ID:
            continue

        ax, ay = ar['center']
        best_person = None
        best_area = float('inf')

        for p in person_results:
            x1, y1, x2, y2 = p['bbox']
            # Cek apakah ArUco center ada di dalam bbox person
            if x1 <= ax <= x2 and y1 <= ay <= y2:
                area = (x2 - x1) * (y2 - y1)
                # Ambil bbox terkecil yang mengandung marker (paling akurat)
                if area < best_area:
                    best_area = area
                    best_person = p

        if best_person is not None:
            matches.append({
                'person': best_person,
                'aruco': ar,
            })

    return matches


class OperatorTracker:
    """
    Menyimpan memori operator. Jika ArUco hilang selama beberapa frame (flicker/blur),
    tracker ini akan tetap menempelkan identitas ke person bbox terdekat dari posisi terakhir.
    """
    def __init__(self, max_lost_frames=15, distance_threshold=250):
        self.max_lost_frames = max_lost_frames
        self.distance_threshold = distance_threshold
        self.active_operators = {}

    def update(self, persons, aruco_results):
        current_matches = match_aruco_to_person(aruco_results, persons)
        
        matched_aruco_ids = set()
        matched_person_bboxes = set(tuple(m['person']['bbox']) for m in current_matches)
        
        # 1. Update berdasarkan ArUco yang aktif di frame ini
        for match in current_matches:
            aid = match['aruco']['id']
            self.active_operators[aid] = {
                'person': match['person'],
                'aruco': match['aruco'],
                'lost_frames': 0
            }
            matched_aruco_ids.add(aid)
            
        # 2. Tracking operator lama jika ArUco-nya hilang (Flicker compensation)
        keys_to_remove = []
        for aid, data in self.active_operators.items():
            if aid in matched_aruco_ids:
                continue
                
            data['lost_frames'] += 1
            if data['lost_frames'] > self.max_lost_frames:
                keys_to_remove.append(aid)
                continue
                
            last_bbox = data['person']['bbox']
            last_cx = (last_bbox[0] + last_bbox[2]) // 2
            last_cy = (last_bbox[1] + last_bbox[3]) // 2
            
            best_person = None
            best_dist = float('inf')
            
            # Cari YOLO Person terdekat dari posisi operator di frame sebelumnya
            for p in persons:
                p_bbox = tuple(p['bbox'])
                if p_bbox in matched_person_bboxes:
                    continue
                    
                pcx = (p_bbox[0] + p_bbox[2]) // 2
                pcy = (p_bbox[1] + p_bbox[3]) // 2
                
                dist = ((pcx - last_cx)**2 + (pcy - last_cy)**2) ** 0.5
                if dist < best_dist and dist < self.distance_threshold:
                    best_dist = dist
                    best_person = p
            
            if best_person is not None:
                data['person'] = best_person
                matched_person_bboxes.add(tuple(best_person['bbox']))

        for aid in keys_to_remove:
            del self.active_operators[aid]
            
        output = []
        for aid, data in self.active_operators.items():
            output.append({
                'person': data['person'],
                'aruco': data['aruco'],
                'is_lost': data['lost_frames'] > 0
            })
        return output


# ============================================================
# CROPPED POSE ESTIMATION
# ============================================================
class CroppedPoseEstimator:
    """
    MediaPipe Holistic yang dijalankan pada cropped region saja.
    Ini memastikan skeleton hanya di-track untuk orang yang ada di crop.
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Lightest model untuk performa
        )
        self.last_results = None

    def process_crop(self, frame, bbox, padding=CROP_PADDING):
        """
        Jalankan MediaPipe Holistic pada cropped region.

        Args:
            frame: full frame BGR
            bbox: (x1, y1, x2, y2) dari YOLO
            padding: tambahan piksel di sekeliling crop

        Returns:
            results: MediaPipe results (pose_landmarks, dll)
            crop_info: (cx1, cy1, cx2, cy2, crop_w, crop_h) untuk remapping
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Tambah padding
        cx1 = max(0, x1 - padding)
        cy1 = max(0, y1 - padding)
        cx2 = min(w, x2 + padding)
        cy2 = min(h, y2 + padding)

        crop = frame[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]

        if crop_w < 10 or crop_h < 10:
            return None, None

        # Convert BGR → RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb_crop.flags.writeable = False

        # Inference
        results = self.holistic.process(rgb_crop)
        self.last_results = results

        crop_info = (cx1, cy1, cx2, cy2, crop_w, crop_h)
        return results, crop_info

    def draw_skeleton(self, frame, results, crop_info, style="full",
                      draw_body=True, draw_hands=True, draw_face=True):
        """
        Draw skeleton pada frame asli dengan remapping dari crop coordinates.
        Bisa toggle per-feature: body, hands, face.
        """
        if results is None or crop_info is None:
            return

        cx1, cy1, cx2, cy2, crop_w, crop_h = crop_info

        # Draw pose landmarks (remap dari crop ke frame asli)
        if draw_body and results.pose_landmarks:
            self._draw_pose_remapped(frame, results.pose_landmarks,
                                     cx1, cy1, crop_w, crop_h, style)

        # Draw hands (remap)
        if draw_hands:
            if results.right_hand_landmarks:
                self._draw_hand_remapped(frame, results.right_hand_landmarks,
                                         cx1, cy1, crop_w, crop_h, (80, 22, 10), (80, 44, 121))

            if results.left_hand_landmarks:
                self._draw_hand_remapped(frame, results.left_hand_landmarks,
                                         cx1, cy1, crop_w, crop_h, (121, 22, 76), (121, 44, 250))

        # Draw face mesh (remap)
        if draw_face and results.face_landmarks:
            self._draw_face_remapped(frame, results.face_landmarks,
                                      cx1, cy1, crop_w, crop_h)

    def _remap_lm(self, lm, cx1, cy1, crop_w, crop_h):
        """Convert normalized landmark → absolute pixel coords di frame asli."""
        abs_x = int(lm.x * crop_w + cx1)
        abs_y = int(lm.y * crop_h + cy1)
        return abs_x, abs_y

    def _draw_pose_remapped(self, frame, landmarks, cx1, cy1, crop_w, crop_h, style):
        """Draw pose skeleton dengan remapping."""
        mp_holistic = self.mp_holistic
        lms = landmarks.landmark

        if style == "minimal":
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24),
                (23, 25), (25, 27), (24, 26), (26, 28),
            ]
            key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        else:
            connections = list(mp_holistic.POSE_CONNECTIONS)
            key_indices = list(range(33))

        # Draw connections
        for s, e in connections:
            if s >= len(lms) or e >= len(lms):
                continue
            s_lm, e_lm = lms[s], lms[e]
            if (getattr(s_lm, 'visibility', 0) < 0.5 or
                getattr(e_lm, 'visibility', 0) < 0.5):
                continue
            sp = self._remap_lm(s_lm, cx1, cy1, crop_w, crop_h)
            ep = self._remap_lm(e_lm, cx1, cy1, crop_w, crop_h)
            cv2.line(frame, sp, ep, SKELETON_COLOR_CN, 2, cv2.LINE_AA)

        # Draw landmarks
        for idx in key_indices:
            if idx >= len(lms):
                continue
            lm = lms[idx]
            if getattr(lm, 'visibility', 0) < 0.5:
                continue
            pt = self._remap_lm(lm, cx1, cy1, crop_w, crop_h)
            cv2.circle(frame, pt, 4, SKELETON_COLOR_LM, -1)
            cv2.circle(frame, pt, 4, (255, 255, 255), 1)

    def _draw_hand_remapped(self, frame, hand_landmarks, cx1, cy1, crop_w, crop_h,
                            lm_color, cn_color):
        """Draw hand landmarks dengan remapping."""
        mp_holistic = self.mp_holistic
        lms = hand_landmarks.landmark

        for s, e in mp_holistic.HAND_CONNECTIONS:
            if s >= len(lms) or e >= len(lms):
                continue
            sp = self._remap_lm(lms[s], cx1, cy1, crop_w, crop_h)
            ep = self._remap_lm(lms[e], cx1, cy1, crop_w, crop_h)
            cv2.line(frame, sp, ep, cn_color, 2, cv2.LINE_AA)

        for lm in lms:
            pt = self._remap_lm(lm, cx1, cy1, crop_w, crop_h)
            cv2.circle(frame, pt, 3, lm_color, -1)

    def _draw_face_remapped(self, frame, face_landmarks, cx1, cy1, crop_w, crop_h):
        """Draw face mesh landmarks dengan remapping."""
        lms = face_landmarks.landmark
        face_color = (200, 200, 0)  # Cyan-ish
        face_conn_color = (150, 150, 0)

        # Draw face mesh connections (tesselation)
        FACE_CONNECTIONS = self.mp_face_mesh.FACEMESH_TESSELATION
        for s, e in FACE_CONNECTIONS:
            if s >= len(lms) or e >= len(lms):
                continue
            sp = self._remap_lm(lms[s], cx1, cy1, crop_w, crop_h)
            ep = self._remap_lm(lms[e], cx1, cy1, crop_w, crop_h)
            cv2.line(frame, sp, ep, face_conn_color, 1, cv2.LINE_AA)

        # Draw key face landmark points (sparser for performance)
        # Eyes, nose, mouth outline
        KEY_FACE_IDX = [
            # Right eye
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            # Left eye
            362, 382, 381, 380, 374, 373, 390, 249, 263,
            # Nose
            1, 2, 98, 327,
            # Mouth
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        for idx in KEY_FACE_IDX:
            if idx >= len(lms):
                continue
            pt = self._remap_lm(lms[idx], cx1, cy1, crop_w, crop_h)
            cv2.circle(frame, pt, 1, face_color, -1)

    def close(self):
        self.holistic.close()


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_person_bbox(frame, person, is_operator=False, operator_name=""):
    """Draw YOLO person bounding box."""
    x1, y1, x2, y2 = person['bbox']
    conf = person['conf']

    if is_operator:
        color = OPERATOR_BBOX_COLOR
        thickness = 3
        label = f"{operator_name} ({conf:.0%})"
    else:
        color = YOLO_BBOX_COLOR
        thickness = 1
        label = f"person ({conf:.0%})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 if is_operator else 0.45
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
    ly = max(y1 - 8, th + 5)
    cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 8, ly + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, ly), font, font_scale,
                (255, 255, 255) if is_operator else (0, 0, 0), 2 if is_operator else 1, cv2.LINE_AA)

    # Corner accents untuk operator
    if is_operator:
        cl = min(25, (x2 - x1) // 4)
        ct = 4
        for cx, cy, dx, dy in [
            (x1, y1, cl, cl), (x2, y1, -cl, cl),
            (x1, y2, cl, -cl), (x2, y2, -cl, -cl),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)


def draw_aruco_marker(frame, aruco_result):
    """Draw ArUco marker outline."""
    pts = aruco_result['corners'].astype(int)
    cv2.polylines(frame, [pts], True, ARUCO_COLOR, 2, cv2.LINE_AA)
    cx, cy = aruco_result['center']
    cv2.circle(frame, (cx, cy), 4, ARUCO_COLOR, -1)

    # ID label kecil
    id_text = f"ArUco #{aruco_result['id']}"
    cv2.putText(frame, id_text, (cx - 30, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARUCO_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, num_persons, num_markers, operators,
             pose_enabled, pose_style, body_on, hands_on, face_on):
    """Draw HUD."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa = cv2.LINE_AA

    # HUD background
    hud_h = 240 + len(operators) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, max(240, hud_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 35
    # Title
    cv2.putText(frame, "T-MIND Operator Tracking", (20, y), font, 0.6, (200, 220, 255), 2, aa)
    y += 28

    # FPS
    fps_col = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), font, 0.6, fps_col, 2, aa)
    y += 22

    # YOLO info
    cv2.putText(frame, f"Persons (YOLO): {num_persons}", (20, y), font, 0.5, (255, 200, 150), 1, aa)
    y += 20

    # ArUco info
    cv2.putText(frame, f"ArUco Markers: {num_markers}", (20, y), font, 0.5, (0, 255, 255), 1, aa)
    y += 20

    # Pose info
    ps = f"ON ({pose_style})" if pose_enabled else "OFF"
    pc = (0, 255, 200) if pose_enabled else (100, 100, 100)
    cv2.putText(frame, f"Pose: {ps}", (20, y), font, 0.5, pc, 1, aa)
    y += 20

    # Individual toggle status
    def _toggle_text(label, key, enabled):
        status = "ON" if enabled else "OFF"
        color = (0, 230, 180) if enabled else (80, 80, 80)
        return f"  [{key}] {label}: {status}", color

    for label, key, enabled in [("Body", "1", body_on),
                                 ("Hands", "2", hands_on),
                                 ("Face", "3", face_on)]:
        txt, col = _toggle_text(label, key, enabled)
        cv2.putText(frame, txt, (20, y), font, 0.45, col, 1, aa)
        y += 18

    y += 4

    # Operator list
    if operators:
        cv2.putText(frame, "Identified Operators:", (20, y), font, 0.5, (180, 255, 180), 1, aa)
        y += 18
        for op in operators:
            op_text = f"  ID{op['aruco']['id']}: {op['aruco']['name']}"
            cv2.putText(frame, op_text, (20, y), font, 0.5, (100, 255, 150), 1, aa)
            y += 18
    else:
        cv2.putText(frame, "No operator identified", (20, y), font, 0.5, (100, 100, 100), 1, aa)
        y += 18

    # Instructions
    cv2.putText(frame, "'q'=Quit 'p'=Pose 's'=Style '1'=Body '2'=Hands '3'=Face",
                (w - 520, h - 15), font, 0.42, (150, 150, 150), 1, aa)


# ============================================================
# MAIN
# ============================================================
def main():
    global POSE_DRAW_STYLE, ENABLE_BODY_POSE, ENABLE_HANDS, ENABLE_FACE

    print("=" * 60)
    print("  T-MIND: Operator Tracking with ArUco + YOLO + Pose")
    print("=" * 60)

    # Initialize detectors
    print("\n[INFO] Initializing...")
    aruco_detector = ArUcoDetector(ARUCO_DICT_TYPE)
    yolo_detector = YOLOPersonDetector(YOLO_MODEL, YOLO_CONF)
    pose_estimator = CroppedPoseEstimator()
    operator_tracker = OperatorTracker(max_lost_frames=15, distance_threshold=250)

    # Open camera
    print(f"[INFO] Membuka kamera (index: {CAMERA_INDEX}) dengan Multi-threading...")
    cap = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.ret:
        print("[ERROR] Tidak dapat membuka kamera!")
        sys.exit(1)
    cap.start()

    print(f"\n[INFO] Konfigurasi:")
    print(f"  YOLO Model: {YOLO_MODEL} | Conf: {YOLO_CONF}")
    print(f"  ArUco Dict: {ARUCO_DICT_TYPE}")
    print(f"  Target Operator: {TARGET_OPERATOR_ID if TARGET_OPERATOR_ID is not None else 'Semua'}")
    print(f"  Pose: {'ON' if ENABLE_POSE else 'OFF'} | Style: {POSE_DRAW_STYLE}")
    print(f"  Body: {'ON' if ENABLE_BODY_POSE else 'OFF'} | Hands: {'ON' if ENABLE_HANDS else 'OFF'} | Face: {'ON' if ENABLE_FACE else 'OFF'}")
    print(f"\n[INFO] Pipeline: YOLO→ArUco→Match→Crop→Pose")
    print(f"  'q'/ESC = Keluar | 'p' = Toggle Pose | 's' = Style")
    print(f"  '1' = Toggle Body | '2' = Toggle Hands | '3' = Toggle Face")
    print(f"\n[INFO] Deteksi dimulai!\n")

    pose_enabled = ENABLE_POSE
    body_on = ENABLE_BODY_POSE
    hands_on = ENABLE_HANDS
    face_on = ENABLE_FACE
    pose_styles = ["full", "minimal"]
    pose_si = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0

    fps = 0
    frame_count = 0
    start_time = time.time()

    try:
        while not cap.stopped:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Beri jeda kecil agar loop tidak membuat CPU 100% jika frame belum ready
                time.sleep(0.005)
                continue

            # JANGAN flip — ArUco marker tidak simetris
            # frame = cv2.flip(frame, 1)

            # ========================================
            # STEP 1: YOLO Person Detection
            # ========================================
            persons = yolo_detector.detect_persons(frame)

            # ========================================
            # STEP 2: ArUco Marker Detection
            # ========================================
            aruco_results = aruco_detector.detect(frame)

            # ========================================
            # STEP 3: Match ArUco → YOLO Person dengan Tracking Memory
            # ========================================
            operators = operator_tracker.update(persons, aruco_results)

            # Set of matched person bboxes
            matched_bboxes = set()
            for op in operators:
                matched_bboxes.add(tuple(op['person']['bbox']))

            # ========================================
            # STEP 4: Draw non-operator persons (dimmed)
            # ========================================
            for p in persons:
                is_op = tuple(p['bbox']) in matched_bboxes
                if not is_op:
                    draw_person_bbox(frame, p, is_operator=False)

            # ========================================
            # STEP 5: Process matched operators
            # ========================================
            for op in operators:
                person = op['person']
                ar = op['aruco']
                is_lost = op.get('is_lost', False)

                # Draw operator bbox (highlighted)
                # Tampilkan tulisan (LOST) jika kita menggunakan tracker memori tapi ArUco fisiknya tidak kelihatan
                display_name = ar['name'] if not is_lost else f"{ar['name']} (TRACKING)"
                draw_person_bbox(frame, person, is_operator=True,
                                 operator_name=display_name)

                # Draw ArUco marker HANYA jika fisiknya terdeteksi frame tersebut
                if not is_lost:
                    draw_aruco_marker(frame, ar)

                # Pose estimation pada cropped region
                if pose_enabled:
                    results, crop_info = pose_estimator.process_crop(
                        frame, person['bbox'], padding=CROP_PADDING)
                    pose_estimator.draw_skeleton(
                        frame, results, crop_info, style=POSE_DRAW_STYLE,
                        draw_body=body_on, draw_hands=hands_on,
                        draw_face=face_on)

            # Draw unmatched ArUco markers
            matched_aruco_ids = set(op['aruco']['id'] for op in operators)
            for ar in aruco_results:
                if ar['id'] not in matched_aruco_ids:
                    draw_aruco_marker(frame, ar)

            # ========================================
            # STEP 6: FPS
            # ========================================
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # ========================================
            # STEP 7: HUD
            # ========================================
            draw_hud(frame, fps, len(persons), len(aruco_results),
                     operators, pose_enabled, POSE_DRAW_STYLE,
                     body_on, hands_on, face_on)

            cv2.imshow("T-MIND Operator Tracking | ArUco+YOLO+Pose", frame)

            # Log
            if operators:
                names = ", ".join([f"ID{op['aruco']['id']}={op['aruco']['name']}"
                                   for op in operators])
                print(f"\r[TRACKING] {names}     ", end="", flush=True)

            # ========================================
            # STEP 8: Keyboard
            # ========================================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                pose_enabled = not pose_enabled
                print(f"\n[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")
            elif key == ord('s'):
                pose_si = (pose_si + 1) % len(pose_styles)
                POSE_DRAW_STYLE = pose_styles[pose_si]
                print(f"\n[INFO] Pose Style: {POSE_DRAW_STYLE}")
            elif key == ord('1'):
                body_on = not body_on
                ENABLE_BODY_POSE = body_on
                print(f"\n[INFO] Body Pose: {'ON' if body_on else 'OFF'}")
            elif key == ord('2'):
                hands_on = not hands_on
                ENABLE_HANDS = hands_on
                print(f"\n[INFO] Hands: {'ON' if hands_on else 'OFF'}")
            elif key == ord('3'):
                face_on = not face_on
                ENABLE_FACE = face_on
                print(f"\n[INFO] Face: {'ON' if face_on else 'OFF'}")

    finally:
        pose_estimator.close()
        cap.stop()
        cv2.destroyAllWindows()
        print("\n\n[INFO] Program selesai.")


if __name__ == "__main__":
    main()
