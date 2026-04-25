"""
V2.5 SOP Pose - Simulate Bearing Sequence + Pose Detection - T-MIND Project
=============================================================================
Menggabungkan V2.4 (Milk Box Bearing Sequence + ROI + YOLO Custom) dengan
V2.2 (Person Tracking + ArUco + MediaPipe Holistic Pose Detection).

Fitur Utama:
  1. Deteksi objek dengan custom model 'Training Kotak Susu.pt'
     sekaligus model COCO (yolov8n) secara bersamaan.
  2. User bisa mendefinisikan satu atau beberapa zona ROI (Region of Interest)
     langsung dari kamera dengan klik+drag mouse.
  3. YOLO hanya mendeteksi object DALAM zona ROI yang sudah didefinisikan.
  4. Work Area graphic: L-shape corner markers + reference lines (H/V).
  5. Bisa simpan/load konfigurasi ROI + WorkArea ke file JSON.
  6. Sequence Tracking: Deteksi urutan perakitan (SOP) bearing.
  7. ArUco marker → YOLO person matching → identifikasi operator.
  8. MediaPipe Holistic Pose → skeleton overlay pada operator yang teridentifikasi.

Kontrol:
  --- ROI & Detection ---
  Klik + Drag  = Gambar zona ROI baru
  ENTER        = Toggle SETUP / DETECT mode
  'c'          = Clear semua ROI
  'd'          = Delete ROI terakhir
  's'          = Save konfigurasi ROI + WorkArea ke JSON
  'l'          = Load konfigurasi ROI + WorkArea dari JSON
  'm'          = Ganti containment mode (center / overlap / full)
  'q' / ESC    = Keluar
  'r'          = Reset Sequence Tracker

  --- Pose Detection ---
  'p'          = Toggle Pose ON/OFF (semua)
  'o'          = Toggle Skeleton style (full/minimal)
  '1'          = Toggle Body Pose ON/OFF
  '2'          = Toggle Hands (jari) ON/OFF
  '3'          = Toggle Face Mesh (mata) ON/OFF

  --- Work Area ---
  'w'          = Mode taruh L-corner (klik di frame)
  'h'          = Mode taruh garis referensi Horizontal
  'v'          = Mode taruh garis referensi Vertikal
  'z'          = Kembali ke IDLE (ROI draw mode)
  'x'          = Clear semua Work Area graphics
  'n'          = Undo corner / line terakhir

Mode:
  SETUP   = Gambar ROI dulu (tekan ENTER untuk mulai deteksi)
  DETECT  = Deteksi YOLO dalam ROI + Pose Tracking
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import time
import sys
import json
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy


# ============================================================
# CONFIGURATION — Edit sesuai kebutuhan
# ============================================================

# --- Kamera ---
CAMERA_INDEX  = 1          # 0 = default webcam, 1 = external
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# --- DroidCam (comment-out, aktifkan jika pakai IP cam / DroidCam) ---
#CAMERA_INDEX  = "http://192.168.0.4:4747/video"   # URL DroidCam
#FRAME_WIDTH   = 1280
#FRAME_HEIGHT  = 720

# --- YOLO ---
CUSTOM_MODEL  = "Training Kotak Susu V2.pt"
COCO_MODEL    = "yolov8n.pt"
DETECT_MODE_MODEL = "custom"
YOLO_CONF     = 0.40
YOLO_CLASSES  = None           # None = semua class

# --- YOLO Person Detection (untuk ArUco matching) ---
YOLO_PERSON_MODEL    = "yolov8n.pt"
YOLO_PERSON_CONF     = 0.40
YOLO_PERSON_CLASS    = 0       # Class ID untuk "person" di COCO

# --- ArUco Detection ---
ARUCO_DICT_TYPE = "4x4_50"

# --- Mapping ID → Nama Operator ---
ID_MAP = {
    0: "Arya Dwi Kusuma",
    1: "Person B",
    2: "Person C",
    3: "Person D",
    4: "Person E",
}
DEFAULT_NAME = "Unknown"

# --- Target Operator ---
TARGET_OPERATOR_ID = None     # None = semua, 0 = hanya ID 0, dst.

# --- MediaPipe Pose ---
ENABLE_POSE         = True
ENABLE_BODY_POSE    = True
ENABLE_HANDS        = True
ENABLE_FACE         = True
POSE_DRAW_STYLE     = "full"            # "full" atau "minimal"
CROP_PADDING        = 30

# --- ROI ---
ROI_COLOR          = (0,   230, 255)
ROI_ACTIVE_COLOR   = (0,   255, 100)
ROI_FILL_ALPHA     = 0.12
ROI_SAVE_FILE      = "roi_config.json"

# --- Warna ---
DETECT_COLOR         = (50, 205, 50)
LABEL_BG             = (20, 20, 20)
YOLO_BBOX_COLOR      = (255, 180, 0)
OPERATOR_BBOX_COLOR  = (0, 255, 100)
ARUCO_COLOR          = (0, 255, 255)
NAME_COLOR           = (0, 255, 255)
SKELETON_COLOR_LM    = (245, 117, 66)
SKELETON_COLOR_CN    = (245, 66, 230)

# --- Work Area Graphic ---
WORK_AREA_COLOR       = (255, 200,  50)
WORK_AREA_LINE_COLOR  = (200, 100, 255)
WORK_AREA_L_SIZE      = 30
WORK_AREA_L_THICKNESS = 3
WORK_AREA_LINE_THICK  = 1

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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Resolusi Input Kamera: {self.actual_w}x{self.actual_h}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                return
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            self.ret, self.frame = ret, frame

    def read(self):
        if self.ret and self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()


# ============================================================
# ARUCO DETECTOR (dari V2.2)
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
# YOLO PERSON DETECTOR (dari V2.2 — untuk ArUco matching)
# ============================================================
class YOLOPersonDetector:
    def __init__(self, model_path=YOLO_PERSON_MODEL, conf=YOLO_PERSON_CONF):
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO Person model: {model_path}...")
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
# MATCHING & TRACKING: ArUco → YOLO Person (dari V2.2)
# ============================================================
def match_aruco_to_person(aruco_results, person_results):
    """
    Match setiap ArUco marker ke YOLO person bbox.
    ArUco center harus berada di dalam bbox person.
    """
    matches = []

    for ar in aruco_results:
        if TARGET_OPERATOR_ID is not None and ar['id'] != TARGET_OPERATOR_ID:
            continue

        ax, ay = ar['center']
        best_person = None
        best_area = float('inf')

        for p in person_results:
            x1, y1, x2, y2 = p['bbox']
            if x1 <= ax <= x2 and y1 <= ay <= y2:
                area = (x2 - x1) * (y2 - y1)
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
# CROPPED POSE ESTIMATION (dari V2.2)
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
        face_color = (200, 200, 0)
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
# ROI DATA CLASS (dari V2.4)
# ============================================================
@dataclass
class ROIZone:
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 230, 255))

    def contains_point(self, px: int, py: int) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def contains_bbox(self, bx1: int, by1: int, bx2: int, by2: int,
                      mode: str = "center") -> bool:
        if mode == "center":
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            return self.contains_point(cx, cy)
        elif mode == "overlap":
            return not (bx2 < self.x1 or bx1 > self.x2 or
                        by2 < self.y1 or by1 > self.y2)
        elif mode == "full":
            return (bx1 >= self.x1 and by1 >= self.y1 and
                    bx2 <= self.x2 and by2 <= self.y2)
        return False

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "ROIZone":
        return cls(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                   name=d.get("name", ""))


# ============================================================
# WORK AREA: L-CORNER & REFERENCE LINE DATA CLASSES (dari V2.4)
# ============================================================
@dataclass
class LCorner:
    """Satu titik sudut L-shape untuk marking pojok work area."""
    x: int
    y: int
    orientation: str = "TL"

    def draw(self, frame, color=WORK_AREA_COLOR,
             size=WORK_AREA_L_SIZE, thickness=WORK_AREA_L_THICKNESS):
        x, y = self.x, self.y
        s = size
        dirs = {
            "TL": (+s,  0,  0, +s),
            "TR": (-s,  0,  0, +s),
            "BL": (+s,  0,  0, -s),
            "BR": (-s,  0,  0, -s),
        }.get(self.orientation, (+s, 0, 0, +s))
        dx1, dy1, dx2, dy2 = dirs
        cv2.line(frame, (x, y), (x + dx1, y + dy1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + dx2, y + dy2), color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (x, y), thickness + 1, color, -1, cv2.LINE_AA)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "orientation": self.orientation}

    @classmethod
    def from_dict(cls, d):
        return cls(x=d["x"], y=d["y"], orientation=d.get("orientation", "TL"))


@dataclass
class RefLine:
    """Satu garis referensi (horizontal atau vertical)."""
    axis: str
    pos:  int
    label: str = ""

    def draw(self, frame, color=WORK_AREA_LINE_COLOR,
             thickness=WORK_AREA_LINE_THICK):
        h, w = frame.shape[:2]
        if self.axis == "H":
            cv2.line(frame, (0, self.pos), (w, self.pos),
                     color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"y={self.pos}"
            cv2.putText(frame, lbl, (w - 90, self.pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            cv2.line(frame, (self.pos, 0), (self.pos, h),
                     color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"x={self.pos}"
            cv2.putText(frame, lbl, (self.pos + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def to_dict(self):
        return {"axis": self.axis, "pos": self.pos, "label": self.label}

    @classmethod
    def from_dict(cls, d):
        return cls(axis=d["axis"], pos=d["pos"], label=d.get("label", ""))


# ============================================================
# WORK AREA MANAGER (dari V2.4)
# ============================================================
_CORNER_CYCLE = ["TL", "TR", "BR", "BL"]

class WorkAreaManager:
    def __init__(self):
        self.corners:   List[LCorner] = []
        self.ref_lines: List[RefLine] = []
        self.sub_mode   = "IDLE"
        self._preview   = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._preview = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.sub_mode == "PLACE_CORNER":
                orientation = _CORNER_CYCLE[len(self.corners) % 4]
                self.corners.append(LCorner(x=x, y=y, orientation=orientation))
                print(f"[WorkArea] Corner {orientation} @ ({x},{y})")
            elif self.sub_mode == "PLACE_HLINE":
                lbl = f"H{len([l for l in self.ref_lines if l.axis=='H'])+1}"
                self.ref_lines.append(RefLine(axis="H", pos=y, label=lbl))
                print(f"[WorkArea] H-line '{lbl}' @ y={y}")
            elif self.sub_mode == "PLACE_VLINE":
                lbl = f"V{len([l for l in self.ref_lines if l.axis=='V'])+1}"
                self.ref_lines.append(RefLine(axis="V", pos=x, label=lbl))
                print(f"[WorkArea] V-line '{lbl}' @ x={x}")

    def draw(self, frame):
        for line in self.ref_lines:
            line.draw(frame)
        for corner in self.corners:
            corner.draw(frame)

        if self.sub_mode != "IDLE" and self._preview:
            px, py = self._preview
            h, w   = frame.shape[:2]
            ghost_col = (255, 255, 100)
            overlay = frame.copy()

            if self.sub_mode == "PLACE_CORNER":
                orientation = _CORNER_CYCLE[len(self.corners) % 4]
                LCorner(px, py, orientation).draw(overlay, color=ghost_col, thickness=2)
            elif self.sub_mode == "PLACE_HLINE":
                cv2.line(overlay, (0, py), (w, py), ghost_col, 1, cv2.LINE_AA)
            elif self.sub_mode == "PLACE_VLINE":
                cv2.line(overlay, (px, 0), (px, h), ghost_col, 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            hint_map = {
                "PLACE_CORNER": f"[WorkArea] Klik: taruh corner {_CORNER_CYCLE[len(self.corners) % 4]}",
                "PLACE_HLINE":  "[WorkArea] Klik: taruh H-line",
                "PLACE_VLINE":  "[WorkArea] Klik: taruh V-line",
            }
            cv2.putText(frame, hint_map.get(self.sub_mode, ""),
                        (10, frame.shape[0] - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 240, 80), 1, cv2.LINE_AA)

    def delete_last_corner(self):
        if self.corners:
            c = self.corners.pop()
            print(f"[WorkArea] Corner {c.orientation} dihapus.")
        else:
            print("[WorkArea] Tidak ada corner.")

    def delete_last_line(self):
        if self.ref_lines:
            l = self.ref_lines.pop()
            print(f"[WorkArea] {l.axis}-line '{l.label}' dihapus.")
        else:
            print("[WorkArea] Tidak ada reference line.")

    def clear(self):
        self.corners.clear()
        self.ref_lines.clear()
        print("[WorkArea] Semua graphic dihapus.")

    def save_to_dict(self) -> dict:
        return {
            "corners":   [c.to_dict() for c in self.corners],
            "ref_lines": [l.to_dict() for l in self.ref_lines],
        }

    def load_from_dict(self, d: dict):
        self.corners   = [LCorner.from_dict(c) for c in d.get("corners",   [])]
        self.ref_lines = [RefLine.from_dict(l)  for l in d.get("ref_lines", [])]
        print(f"[WorkArea] Loaded {len(self.corners)} corners, {len(self.ref_lines)} ref lines.")


# ============================================================
# PALETTE WARNA ROI
# ============================================================
ROI_PALETTE = [
    (0,   230, 255),   # Cyan
    (255, 100,  50),   # Orange
    (180,  50, 255),   # Purple
    (50,  255, 150),   # Green
    (255, 200,  30),   # Yellow
    (50,  150, 255),   # Blue
    (255,  50, 150),   # Pink
]


# ============================================================
# YOLO DUAL-MODEL DETECTOR (dari V2.4 — untuk objek/kotak susu)
# ============================================================
class YOLODetector:
    """
    Mendukung deteksi dengan satu atau dua model sekaligus:
      - custom model (Training Kotak Susu.pt)
      - coco model   (yolov8n.pt)
      - atau keduanya (both)
    """

    def __init__(self, detect_mode: str = DETECT_MODE_MODEL,
                 conf: float = YOLO_CONF, classes=YOLO_CLASSES):
        from ultralytics import YOLO

        self.detect_mode = detect_mode
        self.conf        = conf
        self.classes     = classes
        self.model_custom = None
        self.model_coco   = None

        script_dir   = os.path.dirname(os.path.abspath(__file__))
        custom_path  = os.path.join(script_dir, CUSTOM_MODEL)

        if detect_mode in ("custom", "both"):
            if os.path.exists(custom_path):
                print(f"[INFO] Loading custom model: {custom_path}")
                self.model_custom = YOLO(custom_path)
            else:
                print(f"[WARN] Custom model tidak ditemukan: '{custom_path}'")
                print(f"       Pastikan file '{CUSTOM_MODEL}' ada di folder yang sama dengan script ini.")
                if detect_mode == "custom":
                    print("[WARN] Fallback ke yolov8n.pt (COCO)")
                    self.model_coco = YOLO(COCO_MODEL)

        if detect_mode in ("coco", "both") or \
           (detect_mode == "custom" and self.model_custom is None):
            print(f"[INFO] Loading COCO model: {COCO_MODEL}")
            self.model_coco = YOLO(COCO_MODEL)

    def _run_model(self, model, frame) -> list:
        results = model(frame, conf=self.conf, classes=self.classes, verbose=False)
        boxes = []
        names = model.names
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append({
                    "bbox":       (x1, y1, x2, y2),
                    "conf":       float(box.conf[0]),
                    "class_id":   int(box.cls[0]),
                    "class_name": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "source":     "custom" if model is self.model_custom else "coco",
                })
        return boxes

    def detect(self, frame, roi_zones: List[ROIZone],
               iou_mode: str = "center") -> List[dict]:
        if not roi_zones:
            return []

        raw = []
        if self.model_custom:
            raw.extend(self._run_model(self.model_custom, frame))
        if self.model_coco:
            raw.extend(self._run_model(self.model_coco, frame))

        detections = []
        for det in raw:
            x1, y1, x2, y2 = det["bbox"]
            inside_roi = -1
            for i, roi in enumerate(roi_zones):
                if roi.contains_bbox(x1, y1, x2, y2, mode=iou_mode):
                    inside_roi = i
                    break
            if inside_roi >= 0:
                det["roi_idx"] = inside_roi
                detections.append(det)

        return detections


# ============================================================
# SEQUENCE TRACKER (dari V2.4)
# ============================================================
SEQUENCE_STEPS = [
    {"action": "INIT",  "desc": "Siapkan Storage Penuh, Jig Kosong"},
    {"action": "TAKE",  "roi_idx": 0, "desc": "Ambil Storage #1"},
    {"action": "TAKE",  "roi_idx": 2, "desc": "Ambil Storage #3"},
    {"action": "TAKE",  "roi_idx": 4, "desc": "Ambil Storage #5"},
    {"action": "PLACE", "roi_idx": 9, "desc": "Taruh Jig #5 (Kanan)"},
    {"action": "PLACE", "roi_idx": 7, "desc": "Taruh Jig #3 (Tengah)"},
    {"action": "PLACE", "roi_idx": 5, "desc": "Taruh Jig #1 (Kiri)"},
    {"action": "TAKE",  "roi_idx": 1, "desc": "Ambil Storage #2"},
    {"action": "TAKE",  "roi_idx": 3, "desc": "Ambil Storage #4"},
    {"action": "PLACE", "roi_idx": 8, "desc": "Taruh Jig #4"},
    {"action": "PLACE", "roi_idx": 6, "desc": "Taruh Jig #2"},
    {"action": "DONE",  "desc": "Perakitan Selesai!"}
]

def get_roi_name(idx):
    return f"Storage #{idx + 1}" if idx < 5 else f"Jig #{idx - 5 + 1}"

class SequenceTracker:
    def __init__(self, num_rois=10, debounce_frames=5):
        self.num_rois = num_rois
        self.debounce_frames = debounce_frames
        self.roi_history = [0] * num_rois
        self.roi_states = [False] * num_rois
        self.expected_states = [False] * num_rois
        self.current_step = 0
        self.error_msg = ""

    def reset(self):
        self.roi_history = [0] * self.num_rois
        self.roi_states = [False] * self.num_rois
        self.expected_states = [False] * self.num_rois
        self.current_step = 0
        self.error_msg = ""

    def update(self, detected_roi_indices: List[int]):
        for i in range(self.num_rois):
            if i in detected_roi_indices:
                self.roi_history[i] = min(self.roi_history[i] + 1, self.debounce_frames * 2)
            else:
                self.roi_history[i] = max(self.roi_history[i] - 1, 0)

            if self.roi_history[i] >= self.debounce_frames:
                self.roi_states[i] = True
            elif self.roi_history[i] == 0:
                self.roi_states[i] = False

        if self.current_step >= len(SEQUENCE_STEPS):
            self.error_msg = ""
            return

        step = SEQUENCE_STEPS[self.current_step]
        action = step["action"]

        if action == "INIT":
            if all(self.roi_states[0:5]) and not any(self.roi_states[5:10]):
                self.current_step += 1
                self.expected_states = self.roi_states.copy()
                self.error_msg = ""
            else:
                self.error_msg = "Menunggu Storage 1-5 Penuh & Jig 1-5 Kosong"
        elif action == "DONE":
            self.error_msg = ""
        else:
            roi_idx = step.get("roi_idx", -1)
            target_state = (action == "PLACE")

            wrong_actions = []
            for i in range(self.num_rois):
                if i != roi_idx:
                    if self.roi_states[i] != self.expected_states[i]:
                        wrong_actions.append(i)

            if wrong_actions:
                names = [get_roi_name(w) for w in wrong_actions]
                self.error_msg = f"SALAH URUTAN! Cek: {', '.join(names)}"
            else:
                self.error_msg = f"Instruksi: {step['desc']}"
                if self.roi_states[roi_idx] == target_state:
                    self.current_step += 1
                    self.expected_states[roi_idx] = target_state


# ============================================================
# ROI MANAGER (dari V2.4)
# ============================================================
class ROIManager:
    def __init__(self):
        self.zones: List[ROIZone] = []
        self._drawing    = False
        self._start_pt   = None
        self._current_pt = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing    = True
            self._start_pt   = (x, y)
            self._current_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing and self._start_pt:
                self._drawing = False
                x1 = min(self._start_pt[0], x)
                y1 = min(self._start_pt[1], y)
                x2 = max(self._start_pt[0], x)
                y2 = max(self._start_pt[1], y)
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    idx   = len(self.zones)
                    color = ROI_PALETTE[idx % len(ROI_PALETTE)]
                    name  = f"ROI-{idx + 1}"
                    self.zones.append(ROIZone(x1, y1, x2, y2, name=name, color=color))
                    print(f"[ROI] Ditambah: {name} ({x1},{y1})-({x2},{y2})")
                self._start_pt  = None
                self._current_pt = None

    def draw_in_progress(self, frame):
        if self._drawing and self._start_pt and self._current_pt:
            x1 = min(self._start_pt[0], self._current_pt[0])
            y1 = min(self._start_pt[1], self._current_pt[1])
            x2 = max(self._start_pt[0], self._current_pt[0])
            y2 = max(self._start_pt[1], self._current_pt[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_ACTIVE_COLOR, 2)
            w_px = x2 - x1
            h_px = y2 - y1
            cv2.putText(frame, f"{w_px}x{h_px}px",
                        (x1 + 4, y1 - 6 if y1 > 16 else y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_ACTIVE_COLOR, 1, cv2.LINE_AA)

    def draw_zones(self, frame):
        for i, roi in enumerate(self.zones):
            color = roi.color
            overlay = frame.copy()
            cv2.rectangle(overlay, (roi.x1, roi.y1), (roi.x2, roi.y2), color, -1)
            cv2.addWeighted(overlay, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, frame)
            cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 2)
            # Corner decorations
            cl = min(20, (roi.x2 - roi.x1) // 5)
            ct = 3
            for cx, cy, dx, dy in [
                (roi.x1, roi.y1,  cl,  cl),
                (roi.x2, roi.y1, -cl,  cl),
                (roi.x1, roi.y2,  cl, -cl),
                (roi.x2, roi.y2, -cl, -cl),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
                cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)
            # Label
            label = roi.name if roi.name else f"Zone {i+1}"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            lx, ly = roi.x1 + 6, roi.y1 + th + 6
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
            cv2.putText(frame, label, (lx, ly), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    def clear(self):
        self.zones.clear()
        print("[ROI] Semua zona dihapus.")

    def delete_last(self):
        if self.zones:
            print(f"[ROI] Dihapus: {self.zones.pop().name}")
        else:
            print("[ROI] Tidak ada zona untuk dihapus.")

    def save(self, filepath=ROI_SAVE_FILE, work_area_mgr=None):
        data = {"zones": [z.to_dict() for z in self.zones]}
        if work_area_mgr is not None:
            data["work_area"] = work_area_mgr.save_to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ROI] Disimpan ke '{filepath}' ({len(self.zones)} zona)")

    def load(self, filepath=ROI_SAVE_FILE, work_area_mgr=None):
        if not os.path.exists(filepath):
            print(f"[ROI] File '{filepath}' tidak ditemukan.")
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        self.zones.clear()
        for i, d in enumerate(data.get("zones", [])):
            zone = ROIZone.from_dict(d)
            zone.color = ROI_PALETTE[i % len(ROI_PALETTE)]
            self.zones.append(zone)
        if work_area_mgr is not None and "work_area" in data:
            work_area_mgr.load_from_dict(data["work_area"])
        print(f"[ROI] Loaded {len(self.zones)} zona dari '{filepath}'")


# ============================================================
# DRAWING HELPERS
# ============================================================
COCO_COLORS = [
    (255,  56,  56), (255, 157,  51), (255, 112,  31),
    (255, 178,  29), (207, 210,  49), (72,  249, 100),
    (146, 204,  23), (61,  219, 134), (26,  147,  52),
    (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,   69, 147), (100, 115, 255), (0,    24, 236),
    (132,  56, 255), (82,    0, 133), (203,  56, 255),
    (255, 149, 200), (255,  55, 199),
]

CUSTOM_DETECT_COLOR = (0, 220, 255)

def get_class_color(class_id: int, source: str = "coco") -> Tuple[int, int, int]:
    if source == "custom":
        custom_palette = [
            (0,  220, 255),
            (255, 80,  80),
            (80, 255,  80),
            (255,180,   0),
            (180, 80, 255),
        ]
        return custom_palette[class_id % len(custom_palette)]
    return COCO_COLORS[class_id % len(COCO_COLORS)]


def draw_detection(frame, det: dict):
    """Draw satu deteksi + label."""
    x1, y1, x2, y2 = det["bbox"]
    conf       = det["conf"]
    class_name = det["class_name"]
    roi_idx    = det["roi_idx"]
    source     = det.get("source", "coco")
    color      = get_class_color(det["class_id"], source)

    thickness = 3 if source == "custom" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    src_tag = "★" if source == "custom" else ""
    label   = f"{src_tag}{class_name} {conf:.0%} [Z{roi_idx+1}]"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    ly = y1 - 6 if y1 > th + 10 else y2 + th + 6
    cv2.rectangle(frame, (x1, ly - th - 3), (x1 + tw + 6, ly + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly),
                font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


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

    id_text = f"ArUco #{aruco_result['id']}"
    cv2.putText(frame, id_text, (cx - 30, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARUCO_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, mode: str, roi_count: int,
             det_count: int, containment_mode: str,
             detect_mode_model: str, wa_sub_mode: str,
             seq_tracker=None,
             # Pose info
             num_persons=0, num_markers=0, operators=None,
             pose_enabled=False, pose_style="full",
             body_on=True, hands_on=True, face_on=True):
    """Draw HUD info panel — menggabungkan V2.4 HUD + V2.2 pose info."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa   = cv2.LINE_AA

    if operators is None:
        operators = []

    # Hitung tinggi HUD berdasarkan konten
    hud_h = 320 + len(operators) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, max(320, hud_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 36
    cv2.putText(frame, "T-MIND | SOP Pose + Bearing Sequence", (20, y),
                font, 0.55, (200, 220, 255), 2, aa)
    y += 26

    fps_col = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 80, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), font, 0.55, fps_col, 2, aa)
    y += 22

    mode_col = (0, 255, 100) if mode == "DETECT" else (255, 180, 50)
    cv2.putText(frame, f"Mode: {mode}", (20, y), font, 0.55, mode_col, 2, aa)
    y += 22

    # Model aktif
    model_label = {
        "custom": f"Model: {CUSTOM_MODEL}",
        "coco":   f"Model: {COCO_MODEL}",
        "both":   f"Model: CUSTOM + COCO",
    }.get(detect_mode_model, "Model: ?")
    cv2.putText(frame, model_label, (20, y), font, 0.42, (200, 200, 80), 1, aa)
    y += 20

    cv2.putText(frame, f"ROI Zones: {roi_count}  |  Detections: {det_count}", (20, y),
                font, 0.42, (0, 230, 255), 1, aa)
    y += 20

    cv2.putText(frame, f"Containment: {containment_mode}", (20, y),
                font, 0.42, (180, 180, 180), 1, aa)
    y += 20

    # ---- Pose section ----
    cv2.putText(frame, "--- Pose Detection ---", (20, y), font, 0.42, (200, 180, 255), 1, aa)
    y += 18

    cv2.putText(frame, f"Persons (YOLO): {num_persons}  |  ArUco: {num_markers}",
                (20, y), font, 0.42, (255, 200, 150), 1, aa)
    y += 18

    ps = f"ON ({pose_style})" if pose_enabled else "OFF"
    pc = (0, 255, 200) if pose_enabled else (100, 100, 100)
    cv2.putText(frame, f"Pose: {ps}", (20, y), font, 0.42, pc, 1, aa)
    y += 18

    # Individual toggle status
    def _toggle_text(label, key, enabled):
        status = "ON" if enabled else "OFF"
        color = (0, 230, 180) if enabled else (80, 80, 80)
        return f"  [{key}] {label}: {status}", color

    for label, key, enabled in [("Body", "1", body_on),
                                 ("Hands", "2", hands_on),
                                 ("Face", "3", face_on)]:
        txt, col = _toggle_text(label, key, enabled)
        cv2.putText(frame, txt, (20, y), font, 0.40, col, 1, aa)
        y += 16

    # Operator list
    y += 4
    if operators:
        cv2.putText(frame, "Identified Operators:", (20, y), font, 0.42, (180, 255, 180), 1, aa)
        y += 18
        for op in operators:
            is_lost = op.get('is_lost', False)
            tag = " (TRACKING)" if is_lost else ""
            op_text = f"  ID{op['aruco']['id']}: {op['aruco']['name']}{tag}"
            cv2.putText(frame, op_text, (20, y), font, 0.42, (100, 255, 150), 1, aa)
            y += 18
    else:
        cv2.putText(frame, "No operator identified", (20, y), font, 0.42, (100, 100, 100), 1, aa)
        y += 18

    # WorkArea sub-mode indicator
    if wa_sub_mode != "IDLE":
        wa_col = (255, 240, 80)
        cv2.putText(frame, f"[WorkArea] {wa_sub_mode}", (20, y),
                    font, 0.42, wa_col, 1, aa)

    # Instruksi bawah
    instructions = [
        "Klik+Drag=ROI  ENTER=Start/Stop  C=Clear  D=Del",
        "S=Save  L=Load  Q=Quit  M=Containment  R=Reset Seq",
        "P=Pose  O=Style  1=Body  2=Hands  3=Face",
        "W=Corner  H=Hline  V=Vline  Z=Idle  X=ClearWA  N=Undo",
    ]
    iy = h - len(instructions) * 18 - 8
    for line in instructions:
        cv2.putText(frame, line, (w - 420, iy),
                    font, 0.36, (140, 140, 140), 1, aa)
        iy += 18

    # Sequence Status Panel (Top Right)
    if seq_tracker and roi_count == 10:
        msg = seq_tracker.error_msg
        if "SALAH" in msg:
            color = (50, 50, 255)
        elif "Menunggu" in msg:
            color = (0, 200, 255)
        elif "Perakitan Selesai" in msg:
            color = (50, 255, 50)
        else:
            color = (255, 255, 50)

        box_w, box_h = 450, 70
        bx = w - box_w - 20
        by = 20

        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), color, 2)

        step_text = f"Step {seq_tracker.current_step}/{len(SEQUENCE_STEPS)}" \
                    if seq_tracker.current_step < len(SEQUENCE_STEPS) else "DONE"
        cv2.putText(frame, "SEQUENCE STATUS - " + step_text, (bx + 10, by + 22),
                    font, 0.45, (200, 200, 200), 1, aa)

        (tw, th), _ = cv2.getTextSize(msg, font, 0.55, 2)
        display_msg = msg if tw <= box_w - 20 else msg[:45] + "..."
        cv2.putText(frame, display_msg, (bx + 10, by + 50),
                    font, 0.55, color, 2, aa)


# ============================================================
# MAIN
# ============================================================
def main():
    global POSE_DRAW_STYLE, ENABLE_BODY_POSE, ENABLE_HANDS, ENABLE_FACE

    print("=" * 65)
    print("  T-MIND: SOP Pose + Simulate Bearing Sequence")
    print("  (V2.4 Bearing Sequence + V2.2 Pose Detection)")
    print("=" * 65)
    print()
    print(f"  Detect mode  : {DETECT_MODE_MODEL}")
    print(f"  Custom model : {CUSTOM_MODEL}")
    print(f"  COCO model   : {COCO_MODEL}")
    print(f"  ArUco Dict   : {ARUCO_DICT_TYPE}")
    print(f"  Target Op    : {TARGET_OPERATOR_ID if TARGET_OPERATOR_ID is not None else 'Semua'}")
    print(f"  Pose         : {'ON' if ENABLE_POSE else 'OFF'} | Style: {POSE_DRAW_STYLE}")
    print(f"  Body: {'ON' if ENABLE_BODY_POSE else 'OFF'} | Hands: {'ON' if ENABLE_HANDS else 'OFF'} | Face: {'ON' if ENABLE_FACE else 'OFF'}")
    print()
    print("  Cara penggunaan:")
    print("  1. Gambar zona ROI dengan klik+drag di window kamera")
    print("  2. Tekan ENTER untuk mulai deteksi YOLO + Pose (hanya dalam ROI)")
    print("  3. Tekan ENTER lagi untuk kembali ke mode ROI")
    print()

    # ---- Init camera ----
    print(f"[INFO] Membuka kamera index={CAMERA_INDEX}...")
    cap = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.ret:
        print("[ERROR] Gagal buka kamera!")
        sys.exit(1)
    cap.start()

    # ---- Init YOLO (object detection in ROI) ----
    yolo = YOLODetector(detect_mode=DETECT_MODE_MODEL,
                        conf=YOLO_CONF, classes=YOLO_CLASSES)

    # ---- Init YOLO Person Detector (untuk ArUco → Person matching) ----
    yolo_person = YOLOPersonDetector(YOLO_PERSON_MODEL, YOLO_PERSON_CONF)

    # ---- Init ArUco Detector ----
    aruco_detector = ArUcoDetector(ARUCO_DICT_TYPE)

    # ---- Init Pose Estimator ----
    pose_estimator = CroppedPoseEstimator()

    # ---- Init Operator Tracker ----
    operator_tracker = OperatorTracker(max_lost_frames=15, distance_threshold=250)

    # ---- Init ROI + WorkArea Manager ----
    roi_mgr = ROIManager()
    wa_mgr  = WorkAreaManager()

    # ---- Init Sequence Tracker ----
    seq_tracker = SequenceTracker(num_rois=10, debounce_frames=8)

    # ---- Window ----
    WIN = "T-MIND | SOP Pose + Bearing Sequence  [ENTER=detect, Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cap.actual_w, cap.actual_h)

    def combined_mouse_cb(event, x, y, flags, param):
        if wa_mgr.sub_mode != "IDLE":
            wa_mgr.mouse_callback(event, x, y, flags, param)
        else:
            roi_mgr.mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback(WIN, combined_mouse_cb)

    # ---- State ----
    mode              = "SETUP"
    containment_modes = ["center", "overlap", "full"]
    contain_idx       = 0
    detections        = []
    fps               = 0.0
    frame_count       = 0
    t_start           = time.time()

    pose_enabled = ENABLE_POSE
    body_on = ENABLE_BODY_POSE
    hands_on = ENABLE_HANDS
    face_on = ENABLE_FACE
    pose_styles = ["full", "minimal"]
    pose_si = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0

    # ---- Load config sebelumnya ----
    roi_mgr.load(ROI_SAVE_FILE, work_area_mgr=wa_mgr)
    print(f"\n[INFO] SETUP MODE – gambar zona ROI, lalu tekan ENTER")
    print(f"[INFO] Pipeline: YOLO(ROI) + YOLO(Person)→ArUco→Match→Crop→Pose")
    print(f"  'q'/ESC = Keluar | 'p' = Toggle Pose | 'o' = Style")
    print(f"  '1' = Toggle Body | '2' = Toggle Hands | '3' = Toggle Face\n")

    try:
        while not cap.stopped:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # ========================================
            # STEP 1: YOLO Object Detection (dalam ROI) — dari V2.4
            # ========================================
            if mode == "DETECT" and roi_mgr.zones:
                detections = yolo.detect(
                    frame, roi_mgr.zones,
                    iou_mode=containment_modes[contain_idx]
                )
                # Update Sequence Tracker if we have 10 ROIs
                if len(roi_mgr.zones) == 10:
                    detected_rois = list(set([d["roi_idx"] for d in detections]))
                    seq_tracker.update(detected_rois)
            elif mode == "SETUP":
                detections = []

            # ========================================
            # STEP 2: YOLO Person Detection + ArUco (dari V2.2)
            # ========================================
            persons = yolo_person.detect_persons(frame)
            aruco_results = aruco_detector.detect(frame)

            # ========================================
            # STEP 3: Match ArUco → YOLO Person dengan Tracking Memory
            # ========================================
            operators = operator_tracker.update(persons, aruco_results)

            matched_bboxes = set()
            for op in operators:
                matched_bboxes.add(tuple(op['person']['bbox']))

            # ========================================
            # STEP 4: Draw (layer order)
            # ========================================
            wa_mgr.draw(frame)              # 1. Work area reference
            roi_mgr.draw_zones(frame)       # 2. ROI zones
            roi_mgr.draw_in_progress(frame) # 3. ROI in-drag
            for det in detections:          # 4. Object detections
                draw_detection(frame, det)

            # ========================================
            # STEP 5: Draw non-operator persons (dimmed) — hanya di DETECT mode
            # ========================================
            if mode == "DETECT":
                for p in persons:
                    is_op = tuple(p['bbox']) in matched_bboxes
                    if not is_op:
                        draw_person_bbox(frame, p, is_operator=False)

            # ========================================
            # STEP 6: Process matched operators (bbox + ArUco + Pose)
            # ========================================
            for op in operators:
                person = op['person']
                ar = op['aruco']
                is_lost = op.get('is_lost', False)

                # Draw operator bbox (highlighted)
                display_name = ar['name'] if not is_lost else f"{ar['name']} (TRACKING)"
                draw_person_bbox(frame, person, is_operator=True,
                                 operator_name=display_name)

                # Draw ArUco marker HANYA jika fisiknya terdeteksi
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
            # STEP 7: HUD
            # ========================================
            draw_hud(
                frame, fps, mode,
                roi_count=len(roi_mgr.zones),
                det_count=len(detections),
                containment_mode=containment_modes[contain_idx],
                detect_mode_model=DETECT_MODE_MODEL,
                wa_sub_mode=wa_mgr.sub_mode,
                seq_tracker=seq_tracker,
                # Pose info
                num_persons=len(persons),
                num_markers=len(aruco_results),
                operators=operators,
                pose_enabled=pose_enabled,
                pose_style=POSE_DRAW_STYLE,
                body_on=body_on,
                hands_on=hands_on,
                face_on=face_on,
            )

            # ---- SETUP banner ----
            if mode == "SETUP":
                h, w = frame.shape[:2]
                banner = "SETUP MODE: Gambar ROI, lalu tekan ENTER untuk mulai deteksi"
                (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                bx = (w - bw) // 2
                by = h - 30
                cv2.rectangle(frame, (bx - 8, by - bh - 6),
                              (bx + bw + 8, by + 6), (20, 20, 20), -1)
                cv2.putText(frame, banner, (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 255), 2, cv2.LINE_AA)

            # ---- FPS ----
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

            cv2.imshow(WIN, frame)

            # Log operators
            if operators:
                names = ", ".join([f"ID{op['aruco']['id']}={op['aruco']['name']}"
                                   for op in operators])
                print(f"\r[TRACKING] {names}     ", end="", flush=True)

            # ========================================
            # STEP 8: Keyboard
            # ========================================
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("\n[INFO] Keluar...")
                break

            elif key == 13:                    # ENTER = Toggle mode
                if mode == "SETUP":
                    if not roi_mgr.zones:
                        print("[WARN] Belum ada ROI!")
                    else:
                        mode = "DETECT"
                        if len(roi_mgr.zones) == 10:
                            seq_tracker.reset()
                            print("[INFO] Sequence Tracking aktif (10 ROI Storage & Jig).")
                        print(f"[INFO] DETECT MODE – YOLO aktif dalam {len(roi_mgr.zones)} zona + Pose")
                else:
                    mode = "SETUP"
                    detections = []
                    seq_tracker.reset()
                    print("[INFO] SETUP MODE – edit ROI")

            elif key == ord('r'):
                seq_tracker.reset()
                print("[INFO] Sequence Tracker di-reset.")

            elif key == ord('c'):
                roi_mgr.clear()
                if mode == "DETECT":
                    mode = "SETUP"

            elif key == ord('d'):
                roi_mgr.delete_last()
                if not roi_mgr.zones and mode == "DETECT":
                    mode = "SETUP"

            elif key == ord('s'):
                roi_mgr.save(ROI_SAVE_FILE, work_area_mgr=wa_mgr)

            elif key == ord('l'):
                roi_mgr.load(ROI_SAVE_FILE, work_area_mgr=wa_mgr)

            elif key == ord('m'):
                contain_idx = (contain_idx + 1) % len(containment_modes)
                cmode = containment_modes[contain_idx]
                print(f"[INFO] Containment: '{cmode}'")

            # ---- Pose Controls ----
            elif key == ord('p'):
                pose_enabled = not pose_enabled
                print(f"\n[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")

            elif key == ord('o'):
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

            # ---- Work Area ----
            elif key == ord('w'):
                wa_mgr.sub_mode = "PLACE_CORNER"
                print(f"[WorkArea] PLACE_CORNER – next: {_CORNER_CYCLE[len(wa_mgr.corners) % 4]}")

            elif key == ord('h'):
                wa_mgr.sub_mode = "PLACE_HLINE"
                print("[WorkArea] PLACE_HLINE – klik untuk taruh garis horizontal")

            elif key == ord('v'):
                wa_mgr.sub_mode = "PLACE_VLINE"
                print("[WorkArea] PLACE_VLINE – klik untuk taruh garis vertikal")

            elif key == ord('z'):
                wa_mgr.sub_mode = "IDLE"
                print("[WorkArea] IDLE – kembali ke ROI draw mode")

            elif key == ord('x'):
                wa_mgr.clear()

            elif key == ord('n'):
                if wa_mgr.corners:
                    wa_mgr.delete_last_corner()
                elif wa_mgr.ref_lines:
                    wa_mgr.delete_last_line()

    finally:
        pose_estimator.close()
        cap.stop()
        cv2.destroyAllWindows()
        print("\n\n[INFO] Program selesai.")


# ============================================================
if __name__ == "__main__":
    main()
