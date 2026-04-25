"""
V2.6 SOP Pose - YOLOv8-Pose + Bearing Sequence - T-MIND Project
================================================================
Evolusi dari V2.5: MediaPipe Holistic DIGANTI dengan YOLOv8-Pose
untuk performa FPS yang lebih tinggi.

Keunggulan vs V2.5 (MediaPipe):
  - Satu model (yolov8n-pose) = person detection + 17 keypoints
  - Tidak perlu crop → langsung detect di full frame
  - Multi-person native (semua orang langsung terdeteksi)

Fitur:
  1. YOLOv8-Pose → 17 body keypoints per person (COCO format)
  2. ArUco marker → identifikasi operator
  3. YOLO custom model → deteksi objek (kotak susu) dalam ROI
  4. Sequence Tracker → SOP urutan perakitan bearing
  5. Work Area graphic: L-shape corner markers + reference lines

17 COCO Keypoints:
  0=Nose, 1=L-Eye, 2=R-Eye, 3=L-Ear, 4=R-Ear,
  5=L-Shoulder, 6=R-Shoulder, 7=L-Elbow, 8=R-Elbow,
  9=L-Wrist, 10=R-Wrist, 11=L-Hip, 12=R-Hip,
  13=L-Knee, 14=R-Knee, 15=L-Ankle, 16=R-Ankle

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
  'p'          = Toggle Pose ON/OFF
  'o'          = Toggle Skeleton style (full/minimal)

  --- Work Area ---
  'w'          = Mode taruh L-corner (klik di frame)
  'h'          = Mode taruh garis referensi Horizontal
  'v'          = Mode taruh garis referensi Vertikal
  'z'          = Kembali ke IDLE (ROI draw mode)
  'x'          = Clear semua Work Area graphics
  'n'          = Undo corner / line terakhir
"""

import os
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import sys
import json
import shutil
import threading
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# [OPTIMASI CPU] Maksimalkan penggunaan thread CPU
cv2.setNumThreads(cv2.getNumberOfCPUs())


# ============================================================
# CONFIGURATION
# ============================================================

#-- File Video --- 
#CAMERA_INDEX = r"C:\Users\Victus\Videos\Video Pokeb\Video Testing Dance.mp4"

# --- Kamera ---
CAMERA_INDEX  = 0          # 0 = default webcam, 1 = external

# --- DroidCam (comment-out, aktifkan jika pakai IP cam / DroidCam) ---
#CAMERA_INDEX  = "http://10.156.120.6:4747/video"

# --- Resolusi (hanya berlaku untuk kamera, video file pakai resolusi asli) ---
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# --- YOLO Object Detection (kotak susu, dll) ---
#CUSTOM_MODEL  = "Training bearing dummy V2.pt"
CUSTOM_MODEL  = r"C:\Users\Victus\Downloads\best (9).pt"
#CUSTOM_MODEL  = "Training Kotak Susu V2.pt"
COCO_MODEL    = "yolov8n.pt"
DETECT_MODE_MODEL = "custom"
YOLO_CONF     = 0.40
YOLO_CLASSES  = None

# --- YOLOv8-Pose ---
POSE_MODEL    = "yolov8n-pose.pt"       # Auto-download jika belum ada
POSE_CONF     = 0.40                    # Confidence threshold person+pose
ENABLE_POSE   = True
POSE_DRAW_STYLE = "full"                # "full" atau "minimal"

# --- Hardware Acceleration ---
HARDWARE_TARGET = "cuda"                # Pilihan: "auto", "igpu", "cuda", "cpu"

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

# --- ROI ---
ROI_COLOR          = (0,   230, 255)
ROI_ACTIVE_COLOR   = (0,   255, 100)
ROI_FILL_ALPHA     = 0.12
ROI_SAVE_FILE      = "roi_config.json"

# --- Warna ---
YOLO_BBOX_COLOR      = (255, 180, 0)
OPERATOR_BBOX_COLOR  = (0, 255, 100)
ARUCO_COLOR          = (0, 255, 255)
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

# COCO Skeleton Connections (17 keypoints)
COCO_SKELETON = [
    # Kepala
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Tubuh atas
    (5, 6),                           # bahu kiri-kanan
    (5, 7), (7, 9),                   # lengan kiri
    (6, 8), (8, 10),                  # lengan kanan
    # Tubuh → pinggul
    (5, 11), (6, 12), (11, 12),
    # Kaki
    (11, 13), (13, 15),               # kaki kiri
    (12, 14), (14, 16),               # kaki kanan
]

# Minimal skeleton (tanpa kepala, hanya tubuh & kaki)
COCO_SKELETON_MINIMAL = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COCO_KEYPOINT_NAMES = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
]


# ============================================================
# MULTITHREADING CAMERA / VIDEO FILE
# ============================================================
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        # Deteksi apakah sumber = file video
        self.is_file = isinstance(src, str) and not src.startswith("http")

        if isinstance(src, int) and os.name == 'nt':
            # Hindari freeze webcam eksternal (Logitech, dll) di Windows dengan DirectShow
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.is_file:
            # Hanya set resolusi untuk kamera (file video pakai resolusi asli)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else 0

        src_type = "Video File" if self.is_file else "Kamera"
        print(f"[INFO] {src_type}: {self.actual_w}x{self.actual_h}")
        if self.is_file:
            print(f"[INFO] Video FPS: {self.video_fps:.1f} | Total frames: {self.total_frames}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self._lock = threading.Lock()
        self._src = src

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
                if self.is_file:
                    # Video habis → loop dari awal
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stopped = True
                        return
                    print("\n[INFO] Video selesai, loop dari awal.")
                else:
                    self.stopped = True
                    return
            with self._lock:
                self.ret, self.frame = ret, frame

            # Sinkronisasi FPS untuk file video (agar tidak terlalu cepat)
            if self.is_file and self.video_fps > 0:
                time.sleep(1.0 / self.video_fps)

    def read(self):
        with self._lock:
            if self.ret and self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()


# ============================================================
# ARUCO DETECTOR
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
# YOLOv8-POSE DETECTOR (AUTO GPU DirectML / CUDA)
# ============================================================
class YOLOPoseDetector:
    """
    YOLOv8-Pose: satu model untuk person detection + 17 body keypoints.
    Menggantikan YOLOPersonDetector + CroppedPoseEstimator (MediaPipe).
    GPU Acceleration diaktifkan via DirectML (AMD iGPU) atau CUDA (NVIDIA).
    """

    def __init__(self, model_path=POSE_MODEL, conf=POSE_CONF):
        from ultralytics import YOLO
        import torch
        import os
        import shutil

        self.conf = conf
        self.device = 'cpu'
        self.infer_width = 640 # Resize ke lebar frame ini supaya inference iGPU tidak terlalu berat

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        target = HARDWARE_TARGET.lower()
        if target == "auto":
            if torch.cuda.is_available():
                target = "cuda"
            else:
                try:
                    import onnxruntime as ort
                    if 'DmlExecutionProvider' in ort.get_available_providers():
                        target = "igpu"
                    else:
                        target = "cpu"
                except ImportError:
                    target = "cpu"

        if target == "cuda":
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"[INFO] Loading YOLOv8-Pose model via NVIDIA CUDA...")
                self.model = YOLO(model_path)
            else:
                print(f"[WARN] CUDA tidak tersedia! Fallback ke CPU...")
                self.device = 'cpu'
                self.model = YOLO(model_path)

        elif target == "igpu":
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    self.device = 'dml'
                    onnx_path = f"{model_name}_amd_opt.onnx"
                    if not os.path.exists(onnx_path):
                        print("[INFO] Exporting pose model ke ONNX (FP16 + Simplify untuk AMD iGPU)...")
                        temp_model = YOLO(model_path)
                        exported_file = temp_model.export(
                            format="onnx", imgsz=self.infer_width,
                            half=True, simplify=True
                        )
                        if os.path.exists(exported_file):
                            shutil.move(exported_file, onnx_path)
                        print(f"[INFO] Export ONNX FP16 DirectML selesai: {onnx_path}")
                    
                    self.model = YOLO(onnx_path, task='pose')
                    print(f"[INFO] Loading YOLOv8-Pose model via AMD iGPU (DirectML FP16)...")
                else:
                    raise ImportError("DmlExecutionProvider not found")
            except ImportError as e:
                print(f"[WARN] Gagal init AMD iGPU DirectML! Fallback ke CPU... ({str(e)})")
                self.device = 'cpu'
                self.model = YOLO(model_path)
        else:
            # CPU (atau fallback dr auto)
            self.device = 'cpu'
            print(f"[INFO] Loading YOLOv8-Pose model via CPU...")
            self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect persons + keypoints in one pass.
        Frame di-resize ke ukuran infer_width agar komputasi di iGPU jadi jauh lebih cepat.
        """
        h, w = frame.shape[:2]

        if w > self.infer_width:
            scale = self.infer_width / w
            infer_frame = cv2.resize(frame, (self.infer_width, int(h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            infer_frame = frame

        if self.device == 'dml':
            results = self.model(infer_frame, conf=self.conf, verbose=False)
        elif self.device == 'cuda':
            results = self.model(infer_frame, device='cuda', conf=self.conf, verbose=False)
        else:
            results = self.model(infer_frame, conf=self.conf, verbose=False)

        persons = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes
            keypoints_data = r.keypoints  # YOLOv8-Pose keypoints

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0])

                kpts = None
                if keypoints_data is not None and i < len(keypoints_data):
                    kpts = keypoints_data[i].data[0].cpu().numpy().copy()

                # Remap koordinat balik ke frame asli
                if scale != 1.0:
                    inv_scale = 1.0 / scale
                    x1 = int(x1 * inv_scale)
                    y1 = int(y1 * inv_scale)
                    x2 = int(x2 * inv_scale)
                    y2 = int(y2 * inv_scale)
                    if kpts is not None:
                        kpts[:, 0] *= inv_scale
                        kpts[:, 1] *= inv_scale

                persons.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf_val,
                    'keypoints': kpts,
                })

        return persons


# ============================================================
# MATCHING & TRACKING: ArUco → Person (sama seperti V2.2/V2.5)
# ============================================================
def match_aruco_to_person(aruco_results, person_results):
    """Match setiap ArUco marker ke person bbox terkecil yang mengandung center marker."""
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
            matches.append({'person': best_person, 'aruco': ar})
    return matches


class OperatorTracker:
    """Tracking memory: jaga identitas operator saat ArUco hilang sementara."""

    def __init__(self, max_lost_frames=15, distance_threshold=250):
        self.max_lost_frames = max_lost_frames
        self.distance_threshold = distance_threshold
        self.active_operators = {}

    def update(self, persons, aruco_results):
        current_matches = match_aruco_to_person(aruco_results, persons)
        matched_aruco_ids = set()
        matched_person_bboxes = set(tuple(m['person']['bbox']) for m in current_matches)

        for match in current_matches:
            aid = match['aruco']['id']
            self.active_operators[aid] = {
                'person': match['person'],
                'aruco': match['aruco'],
                'lost_frames': 0
            }
            matched_aruco_ids.add(aid)

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
# ROI DATA CLASS
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
# WORK AREA: L-CORNER & REFERENCE LINE
# ============================================================
@dataclass
class LCorner:
    x: int
    y: int
    orientation: str = "TL"

    def draw(self, frame, color=WORK_AREA_COLOR,
             size=WORK_AREA_L_SIZE, thickness=WORK_AREA_L_THICKNESS):
        x, y = self.x, self.y
        s = size
        dirs = {
            "TL": (+s,  0,  0, +s), "TR": (-s,  0,  0, +s),
            "BL": (+s,  0,  0, -s), "BR": (-s,  0,  0, -s),
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
    axis: str
    pos:  int
    label: str = ""

    def draw(self, frame, color=WORK_AREA_LINE_COLOR,
             thickness=WORK_AREA_LINE_THICK):
        h, w = frame.shape[:2]
        if self.axis == "H":
            cv2.line(frame, (0, self.pos), (w, self.pos), color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"y={self.pos}"
            cv2.putText(frame, lbl, (w - 90, self.pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            cv2.line(frame, (self.pos, 0), (self.pos, h), color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"x={self.pos}"
            cv2.putText(frame, lbl, (self.pos + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def to_dict(self):
        return {"axis": self.axis, "pos": self.pos, "label": self.label}

    @classmethod
    def from_dict(cls, d):
        return cls(axis=d["axis"], pos=d["pos"], label=d.get("label", ""))


# ============================================================
# WORK AREA MANAGER
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
    (0,   230, 255), (255, 100,  50), (180,  50, 255),
    (50,  255, 150), (255, 200,  30), (50,  150, 255), (255,  50, 150),
]


# ============================================================
# YOLO DUAL-MODEL DETECTOR (untuk objek/kotak susu — dari V2.4)
# ============================================================
class YOLODetector:
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
# ROI MANAGER
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
            cl = min(20, (roi.x2 - roi.x1) // 5)
            ct = 3
            for cx, cy, dx, dy in [
                (roi.x1, roi.y1,  cl,  cl), (roi.x2, roi.y1, -cl,  cl),
                (roi.x1, roi.y2,  cl, -cl), (roi.x2, roi.y2, -cl, -cl),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
                cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)
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
            (0,  220, 255), (255, 80,  80), (80, 255,  80),
            (255, 180,  0), (180, 80, 255),
        ]
        return custom_palette[class_id % len(custom_palette)]
    return COCO_COLORS[class_id % len(COCO_COLORS)]


def draw_detection(frame, det: dict):
    """Draw satu deteksi objek + label."""
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
    cv2.putText(frame, label, (x1 + 3, ly), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


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
                (255, 255, 255) if is_operator else (0, 0, 0),
                2 if is_operator else 1, cv2.LINE_AA)

    if is_operator:
        cl = min(25, (x2 - x1) // 4)
        ct = 4
        for cx, cy, dx, dy in [
            (x1, y1, cl, cl), (x2, y1, -cl, cl),
            (x1, y2, cl, -cl), (x2, y2, -cl, -cl),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)


def draw_skeleton(frame, keypoints, style="full",
                  lm_color=SKELETON_COLOR_LM, cn_color=SKELETON_COLOR_CN,
                  kpt_conf_thresh=0.5):
    """
    Draw COCO 17-keypoint skeleton langsung di frame.
    keypoints: np.array shape (17, 3) → [x, y, conf]
    """
    if keypoints is None:
        return

    connections = COCO_SKELETON if style == "full" else COCO_SKELETON_MINIMAL

    # Draw connections
    for s, e in connections:
        if s >= len(keypoints) or e >= len(keypoints):
            continue
        sx, sy, sc = keypoints[s]
        ex, ey, ec = keypoints[e]
        if sc < kpt_conf_thresh or ec < kpt_conf_thresh:
            continue
        cv2.line(frame, (int(sx), int(sy)), (int(ex), int(ey)),
                 cn_color, 2, cv2.LINE_AA)

    # Draw keypoints
    key_indices = list(range(17)) if style == "full" else [5,6,7,8,9,10,11,12,13,14,15,16]
    for idx in key_indices:
        if idx >= len(keypoints):
            continue
        x, y, c = keypoints[idx]
        if c < kpt_conf_thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, lm_color, -1)
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), 1)


def draw_aruco_marker(frame, aruco_result):
    """Draw ArUco marker outline."""
    pts = aruco_result['corners'].astype(int)
    cv2.polylines(frame, [pts], True, ARUCO_COLOR, 2, cv2.LINE_AA)
    cx, cy = aruco_result['center']
    cv2.circle(frame, (cx, cy), 4, ARUCO_COLOR, -1)
    id_text = f"ArUco #{aruco_result['id']}"
    cv2.putText(frame, id_text, (cx - 30, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARUCO_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, mode, roi_count, det_count, containment_mode,
             detect_mode_model, wa_sub_mode, seq_tracker=None,
             num_persons=0, num_markers=0, operators=None,
             pose_enabled=False, pose_style="full"):
    """Draw HUD info panel."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa   = cv2.LINE_AA

    if operators is None:
        operators = []

    hud_h = 290 + len(operators) * 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, max(290, hud_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 36
    cv2.putText(frame, "T-MIND | YOLOv8-Pose + Bearing Seq", (20, y),
                font, 0.55, (200, 220, 255), 2, aa)
    y += 26

    fps_col = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 80, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), font, 0.55, fps_col, 2, aa)
    y += 22

    mode_col = (0, 255, 100) if mode == "DETECT" else (255, 180, 50)
    cv2.putText(frame, f"Mode: {mode}", (20, y), font, 0.55, mode_col, 2, aa)
    y += 22

    model_label = {
        "custom": f"Obj Model: {CUSTOM_MODEL}",
        "coco":   f"Obj Model: {COCO_MODEL}",
        "both":   f"Obj Model: CUSTOM + COCO",
    }.get(detect_mode_model, "Model: ?")
    cv2.putText(frame, model_label, (20, y), font, 0.40, (200, 200, 80), 1, aa)
    y += 18
    cv2.putText(frame, f"Pose Model: {POSE_MODEL}", (20, y), font, 0.40, (200, 200, 80), 1, aa)
    y += 20

    cv2.putText(frame, f"ROI: {roi_count}  |  Obj Det: {det_count}  |  {containment_mode}",
                (20, y), font, 0.40, (0, 230, 255), 1, aa)
    y += 20

    # Pose section
    cv2.putText(frame, "--- Pose (YOLOv8) ---", (20, y), font, 0.40, (200, 180, 255), 1, aa)
    y += 18
    cv2.putText(frame, f"Persons: {num_persons}  |  ArUco: {num_markers}",
                (20, y), font, 0.40, (255, 200, 150), 1, aa)
    y += 18

    ps = f"ON ({pose_style})" if pose_enabled else "OFF"
    pc = (0, 255, 200) if pose_enabled else (100, 100, 100)
    cv2.putText(frame, f"Pose Skeleton: {ps}  ['p'=toggle 'o'=style]", (20, y),
                font, 0.40, pc, 1, aa)
    y += 20

    # Operator list
    if operators:
        cv2.putText(frame, "Identified Operators:", (20, y), font, 0.42, (180, 255, 180), 1, aa)
        y += 18
        for op in operators:
            is_lost = op.get('is_lost', False)
            tag = " (TRACKING)" if is_lost else ""
            op_text = f"  ID{op['aruco']['id']}: {op['aruco']['name']}{tag}"
            cv2.putText(frame, op_text, (20, y), font, 0.40, (100, 255, 150), 1, aa)
            y += 18
    else:
        cv2.putText(frame, "No operator identified", (20, y), font, 0.40, (100, 100, 100), 1, aa)
        y += 18

    if wa_sub_mode != "IDLE":
        cv2.putText(frame, f"[WorkArea] {wa_sub_mode}", (20, y),
                    font, 0.42, (255, 240, 80), 1, aa)

    # Instructions (bawah)
    instructions = [
        "Drag=ROI  ENTER=Start/Stop  C=Clear  D=Del  S=Save  L=Load",
        "P=Pose  O=Style  Q=Quit  M=Containment  R=Reset Seq",
        "W=Corner  H=Hline  V=Vline  Z=Idle  X=ClearWA  N=Undo",
    ]
    iy = h - len(instructions) * 18 - 8
    for line in instructions:
        cv2.putText(frame, line, (w - 450, iy), font, 0.36, (140, 140, 140), 1, aa)
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
        cv2.putText(frame, display_msg, (bx + 10, by + 50), font, 0.55, color, 2, aa)


# ============================================================
# MAIN
# ============================================================
def main():
    global POSE_DRAW_STYLE

    print("=" * 65)
    print("  T-MIND V2.6: YOLOv8-Pose + Bearing Sequence")
    print("  (Pengganti MediaPipe Holistic — FPS lebih tinggi)")
    print("=" * 65)
    print()
    print(f"  Pose model   : {POSE_MODEL} (17 COCO keypoints)")
    print(f"  Obj detect   : {DETECT_MODE_MODEL} ({CUSTOM_MODEL})")
    print(f"  ArUco Dict   : {ARUCO_DICT_TYPE}")
    print(f"  Target Op    : {TARGET_OPERATOR_ID if TARGET_OPERATOR_ID is not None else 'Semua'}")
    print(f"  Pose         : {'ON' if ENABLE_POSE else 'OFF'} | Style: {POSE_DRAW_STYLE}")
    print()

    # ---- Init camera ----
    print(f"[INFO] Membuka kamera index={CAMERA_INDEX}...")
    cap = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.ret:
        print("[ERROR] Gagal buka kamera!")
        sys.exit(1)
    cap.start()

    # ---- Init YOLOv8-Pose (person + keypoints dalam 1 model) ----
    pose_detector = YOLOPoseDetector(POSE_MODEL, POSE_CONF)

    # ---- Init YOLO Object Detector (kotak susu, dll) ----
    yolo_obj = YOLODetector(detect_mode=DETECT_MODE_MODEL,
                            conf=YOLO_CONF, classes=YOLO_CLASSES)

    # ---- Init ArUco ----
    aruco_detector = ArUcoDetector(ARUCO_DICT_TYPE)

    # ---- Init Operator Tracker ----
    operator_tracker = OperatorTracker(max_lost_frames=15, distance_threshold=250)

    # ---- Init ROI + WorkArea + Sequence ----
    roi_mgr = ROIManager()
    wa_mgr  = WorkAreaManager()
    seq_tracker = SequenceTracker(num_rois=10, debounce_frames=8)

    # ---- Window ----
    WIN = "T-MIND V2.6 | YOLOv8-Pose + Bearing Seq  [ENTER=detect, Q=quit]"
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
    pose_styles = ["full", "minimal"]
    pose_si = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0

    # ---- Load config ----
    roi_mgr.load(ROI_SAVE_FILE, work_area_mgr=wa_mgr)
    print(f"\n[INFO] SETUP MODE – gambar zona ROI, lalu tekan ENTER")
    print(f"[INFO] Pipeline: YOLOv8-Pose(Person+Keypoints) + YOLO(ROI Obj) → ArUco → Draw")
    print(f"  'q'/ESC = Keluar | 'p' = Toggle Pose | 'o' = Style\n")

    try:
        while not cap.stopped:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # ========================================
            # STEP 1: Object Detection dalam ROI (kotak susu)
            # ========================================
            if mode == "DETECT" and roi_mgr.zones:
                detections = yolo_obj.detect(
                    frame, roi_mgr.zones,
                    iou_mode=containment_modes[contain_idx]
                )
                if len(roi_mgr.zones) == 10:
                    detected_rois = list(set([d["roi_idx"] for d in detections]))
                    seq_tracker.update(detected_rois)
            elif mode == "SETUP":
                detections = []

            # ========================================
            # STEP 2: YOLOv8-Pose = Person + 17 Keypoints (1 pass!)
            # ========================================
            pose_persons = pose_detector.detect(frame)

            # ========================================
            # STEP 3: ArUco Detection + Match → Person
            # ========================================
            aruco_results = aruco_detector.detect(frame)
            operators = operator_tracker.update(pose_persons, aruco_results)

            matched_bboxes = set()
            for op in operators:
                matched_bboxes.add(tuple(op['person']['bbox']))

            # ========================================
            # STEP 4: Draw layers
            # ========================================
            wa_mgr.draw(frame)
            roi_mgr.draw_zones(frame)
            roi_mgr.draw_in_progress(frame)
            for det in detections:
                draw_detection(frame, det)

            # ========================================
            # STEP 5: Draw non-operator persons (dimmed)
            # ========================================
            if mode == "DETECT":
                for p in pose_persons:
                    is_op = tuple(p['bbox']) in matched_bboxes
                    if not is_op:
                        draw_person_bbox(frame, p, is_operator=False)
                        # Juga gambar skeleton untuk non-operator (dim)
                        if pose_enabled and p['keypoints'] is not None:
                            draw_skeleton(frame, p['keypoints'],
                                          style=POSE_DRAW_STYLE,
                                          lm_color=(120, 80, 50),
                                          cn_color=(120, 50, 100),
                                          kpt_conf_thresh=0.5)

            # ========================================
            # STEP 6: Draw operators (highlighted + skeleton)
            # ========================================
            for op in operators:
                person = op['person']
                ar = op['aruco']
                is_lost = op.get('is_lost', False)

                display_name = ar['name'] if not is_lost else f"{ar['name']} (TRACKING)"
                draw_person_bbox(frame, person, is_operator=True,
                                 operator_name=display_name)

                if not is_lost:
                    draw_aruco_marker(frame, ar)

                # Gambar skeleton operator (warna terang)
                if pose_enabled and person.get('keypoints') is not None:
                    draw_skeleton(frame, person['keypoints'],
                                  style=POSE_DRAW_STYLE,
                                  lm_color=SKELETON_COLOR_LM,
                                  cn_color=SKELETON_COLOR_CN,
                                  kpt_conf_thresh=0.5)

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
                num_persons=len(pose_persons),
                num_markers=len(aruco_results),
                operators=operators,
                pose_enabled=pose_enabled,
                pose_style=POSE_DRAW_STYLE,
            )

            # SETUP banner
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

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

            cv2.imshow(WIN, frame)

            # Log
            if operators:
                names = ", ".join([f"ID{op['aruco']['id']}={op['aruco']['name']}"
                                   for op in operators])
                print(f"\r[TRACKING] {names}     ", end="", flush=True)

            # ========================================
            # STEP 8: Keyboard
            # ========================================
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == 13:  # ENTER
                if mode == "SETUP":
                    if not roi_mgr.zones:
                        print("[WARN] Belum ada ROI!")
                    else:
                        mode = "DETECT"
                        if len(roi_mgr.zones) == 10:
                            seq_tracker.reset()
                            print("[INFO] Sequence Tracking aktif (10 ROI).")
                        print(f"[INFO] DETECT MODE – aktif ({len(roi_mgr.zones)} zona + Pose)")
                else:
                    mode = "SETUP"
                    detections = []
                    seq_tracker.reset()
                    print("[INFO] SETUP MODE")

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
                print(f"[INFO] Containment: '{containment_modes[contain_idx]}'")

            # Pose controls
            elif key == ord('p'):
                pose_enabled = not pose_enabled
                print(f"\n[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")

            elif key == ord('o'):
                pose_si = (pose_si + 1) % len(pose_styles)
                POSE_DRAW_STYLE = pose_styles[pose_si]
                print(f"\n[INFO] Pose Style: {POSE_DRAW_STYLE}")

            # Work Area
            elif key == ord('w'):
                wa_mgr.sub_mode = "PLACE_CORNER"
                print(f"[WorkArea] PLACE_CORNER – next: {_CORNER_CYCLE[len(wa_mgr.corners) % 4]}")
            elif key == ord('h'):
                wa_mgr.sub_mode = "PLACE_HLINE"
                print("[WorkArea] PLACE_HLINE")
            elif key == ord('v'):
                wa_mgr.sub_mode = "PLACE_VLINE"
                print("[WorkArea] PLACE_VLINE")
            elif key == ord('z'):
                wa_mgr.sub_mode = "IDLE"
                print("[WorkArea] IDLE")
            elif key == ord('x'):
                wa_mgr.clear()
            elif key == ord('n'):
                if wa_mgr.corners:
                    wa_mgr.delete_last_corner()
                elif wa_mgr.ref_lines:
                    wa_mgr.delete_last_line()

    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("\n\n[INFO] Program selesai.")


# ============================================================
if __name__ == "__main__":
    main()
