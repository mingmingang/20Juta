"""
V2.1 ArUco Marker Detection + Person Tracking - T-MIND Project
================================================================
Deteksi ArUco marker dari kamera secara real-time.
Setiap marker ID bisa di-mapping ke nama orang.

Kontrol:
  'q' / ESC  = Keluar
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ============================================================
# CONFIGURATION
# ============================================================

# --- Kamera ---
CAMERA_INDEX    = 1                     # Index kamera (0 = default, 1 = external)
FRAME_WIDTH     = 1280                  # Lebar frame
FRAME_HEIGHT    = 720                   # Tinggi frame

# --- ArUco Detection ---
# Dictionary HARUS SAMA dengan yang dipakai saat generate marker!
#   "4x4_50", "4x4_100", "4x4_250"
#   "5x5_50", "5x5_100", "5x5_250"
#   "6x6_50", "6x6_250", "7x7_50"
ARUCO_DICT_TYPE = "4x4_50"              # Harus cocok dengan generator

# --- Mapping ID → Nama ---
# Tambahkan ID dan nama orang di sini
# Format: { marker_id: "Nama Orang" }
ID_MAP = {
    0: "Arya Dwi Kusuma",
    1: "Person B",
    2: "Person C",
    3: "Person D",
    4: "Person E",
    # Tambah sesuai kebutuhan...
}

# Nama default jika ID tidak ada di mapping
DEFAULT_NAME = "Unknown"

# --- Tampilan ---
DRAW_AXIS       = True                  # Gambar garis sumbu di sudut marker
DRAW_ID         = True                  # Tampilkan ID number
DRAW_NAME       = True                  # Tampilkan nama dari ID_MAP
DRAW_BORDER     = True                  # Gambar border di sekeliling marker
BORDER_COLOR    = (0, 255, 0)           # Warna border (BGR) — hijau
NAME_COLOR      = (0, 255, 255)         # Warna nama (BGR) — kuning
ID_COLOR        = (255, 150, 0)         # Warna ID (BGR) — biru muda
BG_COLOR        = (0, 0, 0)             # Warna background teks

# ============================================================

# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "4x4_50":   cv2.aruco.DICT_4X4_50,
    "4x4_100":  cv2.aruco.DICT_4X4_100,
    "4x4_250":  cv2.aruco.DICT_4X4_250,
    "5x5_50":   cv2.aruco.DICT_5X5_50,
    "5x5_100":  cv2.aruco.DICT_5X5_100,
    "5x5_250":  cv2.aruco.DICT_5X5_250,
    "6x6_50":   cv2.aruco.DICT_6X6_50,
    "6x6_250":  cv2.aruco.DICT_6X6_250,
    "7x7_50":   cv2.aruco.DICT_7X7_50,
}


# ============================================================
# ARUCO DETECTOR CLASS
# ============================================================
class ArUcoDetector:
    def __init__(self, dict_type=ARUCO_DICT_TYPE):
        dict_key = ARUCO_DICT_MAP.get(dict_type, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.dict_type = dict_type

    def detect(self, frame):
        """
        Deteksi ArUco markers di frame.
        
        Returns:
            list of dict: [{
                'id': int,
                'name': str,
                'corners': np.array (4 titik sudut),
                'center': (cx, cy),
            }, ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        results = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # 4 corner points
                pts = corners[i][0]  # shape: (4, 2)

                # Center point
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                # Nama dari mapping
                name = ID_MAP.get(int(marker_id), DEFAULT_NAME)

                results.append({
                    'id': int(marker_id),
                    'name': name,
                    'corners': pts,
                    'center': (cx, cy),
                })

        return results

    def draw_results(self, frame, results):
        """Gambar hasil deteksi di frame."""
        for r in results:
            pts = r['corners'].astype(int)
            mid = r['id']
            name = r['name']
            cx, cy = r['center']

            if DRAW_BORDER:
                # Gambar polygon di sekeliling marker
                cv2.polylines(frame, [pts], True, BORDER_COLOR, 3, cv2.LINE_AA)

                # Corner dots
                for j, pt in enumerate(pts):
                    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][j]
                    cv2.circle(frame, tuple(pt), 6, color, -1)

            if DRAW_ID:
                # ID label di atas marker
                id_text = f"ID: {mid}"
                (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tx, ty = cx - tw // 2, pts[0][1] - 15
                cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), BG_COLOR, -1)
                cv2.putText(frame, id_text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ID_COLOR, 2, cv2.LINE_AA)

            if DRAW_NAME:
                # Nama di bawah marker
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx, ty = cx - tw // 2, pts[2][1] + 30
                cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), BG_COLOR, -1)
                cv2.putText(frame, name, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, NAME_COLOR, 2, cv2.LINE_AA)

            # Center dot
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


# ============================================================
# HUD
# ============================================================
def draw_hud(frame, fps, num_detected, results, dict_type):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa = cv2.LINE_AA

    # HUD background
    hud_h = 100 + len(results) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, max(110, hud_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 35
    # FPS
    fps_col = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), font, 0.7, fps_col, 2, aa)
    y += 25

    # Dict info
    cv2.putText(frame, f"Dictionary: {dict_type}", (20, y), font, 0.5, (200, 200, 200), 1, aa)
    y += 20

    # Detection count
    det_col = (0, 255, 200) if num_detected > 0 else (100, 100, 100)
    cv2.putText(frame, f"Markers Detected: {num_detected}", (20, y), font, 0.55, det_col, 1, aa)
    y += 22

    # List detected markers
    for r in results:
        text = f"  ID {r['id']}: {r['name']}"
        cv2.putText(frame, text, (20, y), font, 0.5, (180, 255, 180), 1, aa)
        y += 20

    # Instructions
    cv2.putText(frame, "'q'/ESC = Keluar", (w - 200, h - 15), font, 0.45, (150, 150, 150), 1, aa)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("  ArUco Marker Detection - T-MIND Project")
    print("=" * 55)
    print(f"\n[INFO] Dictionary: {ARUCO_DICT_TYPE}")
    print(f"[INFO] ID Mapping: {ID_MAP}")
    print(f"[INFO] Membuka kamera (index: {CAMERA_INDEX})...\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera!")
        print("[INFO] Coba ganti CAMERA_INDEX (0, 1, 2, ...)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolusi: {actual_w}x{actual_h}")
    print("[INFO] Deteksi dimulai! Arahkan marker ke kamera.\n")

    detector = ArUcoDetector(ARUCO_DICT_TYPE)

    fps = 0
    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Jangan flip! ArUco marker tidak simetris, flip bikin pola berubah
            # frame = cv2.flip(frame, 1)

            # Detect markers
            results = detector.detect(frame)

            # Draw results
            detector.draw_results(frame, results)

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # HUD
            draw_hud(frame, fps, len(results), results, ARUCO_DICT_TYPE)

            # Log detections
            if results:
                names = ", ".join([f"ID{r['id']}={r['name']}" for r in results])
                print(f"\r[DETECTED] {names}     ", end="", flush=True)

            cv2.imshow("ArUco Detection | T-MIND", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n\n[INFO] Program selesai.")


if __name__ == "__main__":
    main()
