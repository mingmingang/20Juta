"""
============================================================
  3.1 MediaPipe Hand Finger Detection + Tray Storage Zones
  ======================================================================
  Sistem:
    - Deteksi jari tangan menggunakan MediaPipe Hand Landmarker
    - Zona Storage berbentuk bebas (Poligon: jajar genjang / trapesium)
    - Terdapat 10 Zona Tray Storage berurutan: 
      (5 Upper Metal Bearing, 5 Lower Metal Bearing)
    - Jika ujung jari menyentuh zona poligon → UI tampilkan
      "Operator mengambil objek di [Nama Zona]"

  Kontrol:
    'q' / ESC  = Keluar
    's'        = Screenshot
    'r'        = Reset semua status
    'e'        = Toggle Edit Mode (Gambar zona manual)
    'm'        = Toggle Mirror Kamera
    MIRROR_CAMERA = True    # mirror gambar (Defaultnya adalah False, ubah ke True jika ingin kameranya ter-mirror secara permanen).
    '[' / ']'  = Pilih Zona Sebelumnya / Selanjutnya (saat Edit Mode)
    ']'        = Lanjut ke tray berikutnya
    '['        = Kembali ke tray sebelumnya
    'c'        = Hapus titik-titik pada zona aktif
    'w'        = Simpan konfigurasi zona ke tray_config.json
    KLIK KIRI  = Tambah titik pojok zona (saat Edit Mode)
    KLIK KANAN = Undo titik terakhir (saat Edit Mode)
============================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import json

# ============================================================
#  KONFIGURASI
# ============================================================

# --- Input Source ---
INPUT_MODE = "camera"   # "camera" = live webcam, "image" = gambar statis
IMAGE_PATH = r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\First Trial (Snitching 3 CV)\Git Clone\20Juta\Trial Fix Arquo\New Bearing Sequence Combination Code\picture pick bearing at storage Mediapipe .png"

# --- Resolusi & Kamera ---
CAMERA_INDEX = 0        # 0 = webcam default, 1 = external
# Jika "auto", akan mengikuti resolusi asli dari gambar statis / default kamera
# Jika diisi angka (misal 1280), akan memaksa (resize) ke resolusi tersebut
FRAME_WIDTH  = "auto"   # Contoh angka: 1280
FRAME_HEIGHT = "auto"   # Contoh angka: 720
MIRROR_CAMERA = False   # True = gambar di-mirror (kiri-kanan)

# --- MediaPipe Hand config ---
HAND_MODEL_PATH      = "hand_landmarker.task"
MAX_NUM_HANDS        = 2
MIN_DETECTION_CONF   = 0.5
MIN_TRACKING_CONF    = 0.5

# --- Polygon Tray Zones ---
ZONES_FILE = "tray_config.json"
ZONE_NAMES = [
    "Upper Metal Bearing 1",
    "Upper Metal Bearing 2",
    "Upper Metal Bearing 3",
    "Upper Metal Bearing 4",
    "Upper Metal Bearing 5",
    "Lower Metal Bearing 1",
    "Lower Metal Bearing 2",
    "Lower Metal Bearing 3",
    "Lower Metal Bearing 4",
    "Lower Metal Bearing 5",
]

# --- Touch Detection ---
TOUCH_RADIUS = 30         # radius (px) untuk deteksi sentuhan fingertip-ke-zona
TOUCH_HOLD_FRAMES = 8     # berapa frame harus touching sebelum trigger

# --- Warna ---
COLOR_BOX_IDLE    = (200, 200, 200)
COLOR_BOX_TOUCHED = (0,   255, 100)
COLOR_HAND_LM     = (0,   200, 255)
COLOR_HAND_CN     = (0,   150, 200)
COLOR_FINGERTIP   = (0,   0,   255)
COLOR_STATUS_BG   = (40,  40,  40)
COLOR_STATUS_OK   = (0,   255, 100)
COLOR_STATUS_WARN = (0,   200, 255)
COLOR_EDIT_MODE   = (0,   0,   255)

# Fingertip landmark indices (MediaPipe Hand: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGERTIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ============================================================
#  MEDIAPIPE HAND LANDMARKER (Tasks API)
# ============================================================
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class HandDetector:
    def __init__(self, model_path=HAND_MODEL_PATH,
                 max_hands=MAX_NUM_HANDS,
                 min_det=MIN_DETECTION_CONF,
                 min_track=MIN_TRACKING_CONF):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path  = os.path.join(script_dir, model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {full_path}\n"
                f"Download: https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=full_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_track,
            min_tracking_confidence=min_track,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0
        print(f"[INFO] HandLandmarker siap (max {max_hands} tangan)")

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts_ms += 33
        results = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        hands = []
        if not results.hand_landmarks:
            return hands

        for i, hand_lm in enumerate(results.hand_landmarks):
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            fingertips = [landmarks[idx] for idx in FINGERTIP_IDS]

            handedness = "Unknown"
            if results.handedness and i < len(results.handedness):
                handedness = results.handedness[i][0].category_name

            hands.append({
                'landmarks': landmarks,
                'fingertips': fingertips,
                'handedness': handedness,
            })
        return hands

    def close(self):
        self.landmarker.close()


# ============================================================
#  INPUT SUMBER: GAMBAR STATIS / KAMERA
# ============================================================
class ImageSource:
    def __init__(self, path, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.frame = cv2.imread(path)
        if self.frame is not None:
            if width != "auto" and height != "auto":
                self.frame = cv2.resize(self.frame, (int(width), int(height)))
            print(f"[INFO] Gambar berhasil dimuat dari: {path} ({self.frame.shape[1]}x{self.frame.shape[0]})")
        else:
            print(f"[ERROR] Gagal memuat gambar dari: {path}")

    def read(self):
        if self.frame is not None:
            # Berikan copy agar tidak tercoret secara permanen
            return True, self.frame.copy()
        return False, None

    def release(self):
        pass

class CameraThread:
    def __init__(self, source, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(source)
        if width != "auto":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height != "auto":
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.running = True
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Kamera terhubung: {source} ({actual_w}x{actual_h})")

    def _grab(self):
        while self.running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# ============================================================
#  TOUCH DETECTION LOGIC
# ============================================================
def point_in_polygon(px, py, points, radius=TOUCH_RADIUS):
    """Cek apakah titik (px,py) berada di dalam atau di dekat batas poligon."""
    if len(points) < 3:
        return False
    contour = np.array(points, dtype=np.int32)
    dist = cv2.pointPolygonTest(contour, (px, py), True)
    return dist >= -radius


def draw_hand_skeleton(frame, hand, draw_fingertips=True):
    landmarks = hand['landmarks']

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),         # Thumb
        (0,5),(5,6),(6,7),(7,8),         # Index
        (5,9),(9,10),(10,11),(11,12),    # Middle
        (9,13),(13,14),(14,15),(15,16),  # Ring
        (13,17),(17,18),(18,19),(19,20), # Pinky
        (0,17),                          # Palm base
    ]

    for (a, b) in CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(frame, landmarks[a], landmarks[b],
                     COLOR_HAND_CN, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (x, y), 3, COLOR_HAND_LM, -1, cv2.LINE_AA)

    if draw_fingertips:
        for idx, fid in enumerate(FINGERTIP_IDS):
            if fid < len(landmarks):
                fx, fy = landmarks[fid]
                cv2.circle(frame, (fx, fy), 8, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                cv2.circle(frame, (fx, fy), 3, COLOR_FINGERTIP, -1, cv2.LINE_AA)


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main():
    print("=" * 55)
    print("  3.1 Hand Finger Detection + Tray Storage Zones")
    print("=" * 55)

    # --- Load Zones ---
    zones = []
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, 'r') as f:
                zones = json.load(f)
            print(f"[INFO] Loaded zones from {ZONES_FILE}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat {ZONES_FILE}: {e}")
            zones = [{"name": name, "points": []} for name in ZONE_NAMES]
    else:
        zones = [{"name": name, "points": []} for name in ZONE_NAMES]
        print(f"[INFO] Membuat file zona baru.")

    # Pastikan jika file JSON kurang dari 10 zone, kita tambahkan sisanya
    loaded_names = [z.get("name") for z in zones]
    for name in ZONE_NAMES:
        if name not in loaded_names:
            zones.append({"name": name, "points": []})

    # --- App State untuk Mouse Callback ---
    app_state = {
        "MODE_EDIT": False,
        "edit_zone_idx": 0,
        "zones": zones,
        "MIRROR_CAMERA": MIRROR_CAMERA
    }

    def mouse_callback(event, x, y, flags, param):
        if not app_state["MODE_EDIT"]:
            return
        idx = app_state["edit_zone_idx"]
        if event == cv2.EVENT_LBUTTONDOWN:
            app_state["zones"][idx]["points"].append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(app_state["zones"][idx]["points"]) > 0:
                app_state["zones"][idx]["points"].pop()

    window_name = "3.1 Hand Detection + Tray Storage"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # --- Init Hand Detector & Input Source ---
    hand_detector = HandDetector()
    
    if INPUT_MODE.lower() == "image":
        cam = ImageSource(IMAGE_PATH)
    else:
        cam = CameraThread(CAMERA_INDEX)
    time.sleep(1.0)

    # --- State Variabel ---
    touch_counters = {}
    touch_triggered = {}
    event_log = []
    MAX_LOG = 5

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    print("\n[INFO] Pipeline aktif!")
    print("  'q'/ESC = Keluar      | 's' = Screenshot | 'r' = Reset")
    print("  'e'     = Edit Mode   | 'w' = Save JSON")
    print("  '['/']' = Prev/Next Zone saat edit | 'c' = Clear points")
    print("  'm'     = Toggle Mirror Kamera")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
            
        if app_state["MIRROR_CAMERA"]:
            frame = cv2.flip(frame, 1)

        h_frame, w_frame = frame.shape[:2]
        output = frame.copy()

        # --- FPS ---
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # ==================================================
        # 1. DETECT HANDS
        # ==================================================
        hands = hand_detector.detect(frame)

        # ==================================================
        # 2. CHECK TOUCH (Jika tidak di mode edit)
        # ==================================================
        touched_boxes = set()

        if not app_state["MODE_EDIT"]:
            for hand in hands:
                for ft_idx, (fx, fy) in enumerate(hand['fingertips']):
                    for box_idx, zone in enumerate(app_state["zones"]):
                        points = zone.get("points", [])
                        if point_in_polygon(fx, fy, points):
                            touched_boxes.add(box_idx)

                            touch_counters[box_idx] = touch_counters.get(box_idx, 0) + 1

                            if (touch_counters[box_idx] >= TOUCH_HOLD_FRAMES and
                                    not touch_triggered.get(box_idx, False)):
                                touch_triggered[box_idx] = True
                                finger_name = FINGERTIP_NAMES[ft_idx]
                                handedness  = hand['handedness']
                                msg = (f"[{time.strftime('%H:%M:%S')}] "
                                       f"Operator mengambil objek di {zone['name']} "
                                       f"({handedness} - {finger_name})")
                                event_log.append(msg)
                                if len(event_log) > MAX_LOG:
                                    event_log.pop(0)
                                print(msg)

                            # Draw touch indicator line
                            contour = np.array(points, dtype=np.int32)
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.line(output, (fx, fy), (cx, cy),
                                         (0, 255, 255), 2, cv2.LINE_AA)

            # Reset counters untuk zona yang tidak tersentuh
            for key in list(touch_counters.keys()):
                if key not in touched_boxes:
                    touch_counters[key] = 0
                    touch_triggered[key] = False

        # ==================================================
        # 3. DRAW POLYGONS (Zona Storage)
        # ==================================================
        for i, zone in enumerate(app_state["zones"]):
            points = zone.get("points", [])
            if len(points) == 0:
                continue

            is_touched = i in touched_boxes
            is_active_edit = app_state["MODE_EDIT"] and (i == app_state["edit_zone_idx"])
            
            color = COLOR_BOX_TOUCHED if is_touched else COLOR_BOX_IDLE
            if is_active_edit:
                color = COLOR_EDIT_MODE

            thickness = 3 if (is_touched or is_active_edit) else 2
            contour = np.array(points, dtype=np.int32)

            # Semi-transparent fill
            overlay = output.copy()
            fill_alpha = 0.4 if is_touched else 0.15
            if is_active_edit:
                fill_alpha = 0.3
            fill_color = COLOR_BOX_TOUCHED if is_touched else (100, 100, 100)
            if is_active_edit:
                fill_color = COLOR_EDIT_MODE

            if len(points) >= 3:
                cv2.fillPoly(overlay, [contour], fill_color)
                cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

            # Border line
            if len(points) >= 2:
                # jika mode edit, biarkan garis tidak tertutup jika masih proses gambar
                # tapi asumsikan selalu tertutup untuk render akhir
                cv2.polylines(output, [contour], isClosed=True, color=color, thickness=thickness)
            
            # Draw individual points saat edit mode
            if is_active_edit:
                for pt in points:
                    cv2.circle(output, tuple(pt), 4, (0, 255, 255), -1)

            # Label text
            label_text = zone["name"]
            if is_touched:
                label_text += " [TOUCHED]"
            if is_active_edit:
                label_text += " [EDITING]"

            # Tempatkan text di titik tertinggi polygon
            tx = min([p[0] for p in points])
            ty = min([p[1] for p in points])
            
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(output, (tx, ty - th - 10), (tx + tw + 8, ty), color, -1)
            cv2.putText(output, label_text, (tx + 4, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # ==================================================
        # 4. DRAW HANDS
        # ==================================================
        for hand in hands:
            draw_hand_skeleton(output, hand)
            if hand['landmarks']:
                wrist = hand['landmarks'][0]
                cv2.putText(output, hand['handedness'],
                            (wrist[0] - 20, wrist[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COLOR_HAND_LM, 2, cv2.LINE_AA)

        # ==================================================
        # 5. UI OVERLAY
        # ==================================================
        panel_h = 100 if app_state["MODE_EDIT"] else 80
        overlay_panel = output.copy()
        cv2.rectangle(overlay_panel, (0, 0), (380, panel_h), COLOR_STATUS_BG, -1)
        cv2.addWeighted(overlay_panel, 0.7, output, 0.3, 0, output)

        cv2.putText(output, f"FPS: {fps_display:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        valid_zones = sum([1 for z in app_state["zones"] if len(z.get("points", [])) >= 3])
        cv2.putText(output, f"Hands: {len(hands)} | Polygons: {valid_zones}/10",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        
        if app_state["MODE_EDIT"]:
            active_z = app_state["zones"][app_state["edit_zone_idx"]]["name"]
            cv2.putText(output, f"EDIT MODE: {active_z}", (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(output, f"Click to draw | '['/']' Next | 'c' Clear | 'w' Save", (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, f"Mode: Detection (Press 'e' to Edit)", (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)

        # Event Log
        if event_log and not app_state["MODE_EDIT"]:
            log_y_start = h_frame - 20 - (len(event_log) * 25)
            overlay_log = output.copy()
            cv2.rectangle(overlay_log, (0, log_y_start - 10),
                          (w_frame, h_frame), COLOR_STATUS_BG, -1)
            cv2.addWeighted(overlay_log, 0.6, output, 0.4, 0, output)

            for j, msg in enumerate(event_log):
                y_pos = log_y_start + j * 25
                color = COLOR_STATUS_OK if j == len(event_log) - 1 else COLOR_STATUS_WARN
                cv2.putText(output, msg, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Big status
        if touched_boxes and not app_state["MODE_EDIT"]:
            status_text = "OPERATOR MENGAMBIL OBJEK DI STORAGE"
            (stw, sth), _ = cv2.getTextSize(status_text,
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            sx = (w_frame - stw) // 2
            sy = 50
            overlay_status = output.copy()
            cv2.rectangle(overlay_status, (sx - 15, sy - sth - 10),
                          (sx + stw + 15, sy + 10), (0, 80, 0), -1)
            cv2.addWeighted(overlay_status, 0.7, output, 0.3, 0, output)
            cv2.putText(output, status_text, (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("[INFO] Keluar...")
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"hand_detect_{ts}.jpg"
            cv2.imwrite(fname, output)
            print(f"[INFO] Screenshot: {fname}")
        elif key == ord('r'):
            touch_counters.clear()
            touch_triggered.clear()
            event_log.clear()
            print("[INFO] Status direset.")
        elif key == ord('e'):
            app_state["MODE_EDIT"] = not app_state["MODE_EDIT"]
            print(f"[INFO] Edit Mode: {app_state['MODE_EDIT']}")
        elif key == ord('w'):
            try:
                with open(ZONES_FILE, 'w') as f:
                    json.dump(app_state["zones"], f, indent=4)
                print(f"[INFO] Zona berhasil disimpan ke {ZONES_FILE}")
            except Exception as e:
                print(f"[ERROR] Gagal menyimpan zona: {e}")
        elif key == ord('c') and app_state["MODE_EDIT"]:
            app_state["zones"][app_state["edit_zone_idx"]]["points"].clear()
            print("[INFO] Zona saat ini dibersihkan.")
        elif key == ord(']') and app_state["MODE_EDIT"]:
            app_state["edit_zone_idx"] = (app_state["edit_zone_idx"] + 1) % len(app_state["zones"])
        elif key == ord('[') and app_state["MODE_EDIT"]:
            app_state["edit_zone_idx"] = (app_state["edit_zone_idx"] - 1) % len(app_state["zones"])
        elif key == ord('m'):
            app_state["MIRROR_CAMERA"] = not app_state["MIRROR_CAMERA"]
            print(f"[INFO] Mirror Kamera: {app_state['MIRROR_CAMERA']}")
            # Otomatis menyesuaikan titik poligon agar tetap di posisi aslinya
            for zone in app_state["zones"]:
                if "points" in zone:
                    for pt in zone["points"]:
                        pt[0] = w_frame - pt[0]

    # --- Cleanup ---
    hand_detector.close()
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
