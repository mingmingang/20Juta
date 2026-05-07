"""
============================================================
  3.0 OCR Trial — YOLO + EasyOCR Real-Time Pipeline
  Target: Bearing shells dengan angka 1-5 di permukaan metalik
  Kamera: Real-time (line camera / USB / IP)
============================================================
"""

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time
import threading
import queue

# ============================================================
#  KONFIGURASI — Sesuaikan dengan setup Anda
# ============================================================

# Mode Deteksi
USE_YOLO        = False  # Set False untuk tes OCR tanpa YOLO (pakai kotak statis di tengah)

# Model YOLO — pakai model custom bearing yang sudah ada
MODEL_PATH = "BEARING 3D PRINT V4.pt"   # ganti sesuai kebutuhan

# Kamera — pilih salah satu:
CAMERA_SOURCE = 0                        # 0 = webcam USB biasa
# CAMERA_SOURCE = "http://192.168.x.x:8080/video"  # IP cam / DroidCam
# CAMERA_SOURCE = 1                      # kamera kedua (line cam)

# Threshold
YOLO_CONF       = 0.35   # confidence YOLO minimum
OCR_CONF        = 0.35   # confidence EasyOCR minimum

# GPU
USE_GPU         = False   # False jika tidak ada GPU
YOLO_DEVICE     = 0      # 0 = GPU pertama (RTX 2050)

# OCR — angka + huruf (alphanumeric)
OCR_ALLOWLIST   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Padding crop sebelum masuk OCR (px)
CROP_PAD        = 15

# Skip frame untuk OCR (OCR berat, tidak perlu tiap frame)
OCR_SKIP_FRAMES = 3      # jalankan OCR setiap N frame

# ============================================================
#  INISIALISASI
# ============================================================

if USE_YOLO:
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    model.to("cuda" if USE_GPU else "cpu")
else:
    model = None
    print("[INFO] YOLO dinonaktifkan. Mode Static Center Crop diaktifkan.")

print("[INFO] Loading EasyOCR... (pertama kali butuh download model ~1.5GB)")
reader = easyocr.Reader(["en"], gpu=USE_GPU, verbose=False)
print("[INFO] EasyOCR siap!")


# ============================================================
#  FUNGSI PRE-PROCESSING UNTUK PERMUKAAN METALIK
# ============================================================

def preprocess_for_metal_ocr(crop):
    """
    Pre-processing khusus permukaan logam berkilap:
    1. CLAHE  - equalize kontras lokal (handle glare)
    2. Sharpen - pertajam tepi angka
    3. Adaptive Threshold - binarisasi adaptif
    4. Tophat morphology - isolasi angka kecil
    """
    # Resize ke ukuran minimal agar OCR akurat (min ~64px tinggi)
    h, w = crop.shape[:2]
    if h < 64:
        scale = 64 / h
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Convert ke grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE — handle refleksi/glare metalik
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gaussian blur ringan (kurangi noise)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 3. Sharpen
    kernel_sharp = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel_sharp)

    # 4. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # 5. Morphological top-hat (isolasi teks kecil di latar tidak rata)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_morph)

    # Gabungkan threshold + tophat
    _, tophat_bin = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(thresh, tophat_bin)

    # Denoise akhir
    denoised = cv2.fastNlMeansDenoising(combined, h=10)

    return denoised, enhanced  # return keduanya untuk debug


# ============================================================
#  FUNGSI OCR PADA CROP
# ============================================================

def run_ocr(crop_img):
    """
    Jalankan EasyOCR pada crop gambar.
    Return list of dict: {text, confidence, bbox}
    """
    processed, enhanced = preprocess_for_metal_ocr(crop_img)

    # Coba pada gambar yang sudah diproses dulu
    results = reader.readtext(
        processed,
        allowlist=OCR_ALLOWLIST,
        detail=1,
        paragraph=False,
        width_ths=0.5,
        height_ths=0.4
    )

    # Jika tidak ada hasil, coba pada enhanced (grayscale CLAHE)
    if not results:
        results = reader.readtext(
            enhanced,
            allowlist=OCR_ALLOWLIST,
            detail=1,
            paragraph=False
        )

    detections = []
    for (bbox, text, conf) in results:
        text = text.strip()
        if conf >= OCR_CONF and text:
            detections.append({
                "text": text,
                "confidence": round(conf, 2)
            })

    return detections


# ============================================================
#  KAMERA THREAD (non-blocking frame grab)
# ============================================================

class CameraThread:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()
        print(f"[INFO] Kamera terhubung: {source}")

    def _grab(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()


# ============================================================
#  PIPELINE UTAMA — REAL-TIME
# ============================================================

def run_realtime():
    cam = CameraThread(CAMERA_SOURCE)
    time.sleep(1.0)  # tunggu kamera siap

    frame_count = 0
    fps_start   = time.time()
    fps         = 0.0

    # Cache hasil OCR (supaya display tetap smooth walau OCR lambat)
    ocr_cache = {}   # key: box_id, value: str teks

    print("[INFO] Pipeline aktif!")
    print("  'q'  = Keluar")
    print("  's'  = Screenshot")
    print("  'd'  = Toggle debug view (pre-processing)")
    print("  'r'  = Reset OCR cache")

    show_debug  = False
    debug_crops = []  # simpan crop untuk debug view

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[WARN] Frame kosong, menunggu...")
            time.sleep(0.05)
            continue

        frame_count += 1
        output = frame.copy()

        # ---- FPS calculation ----
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # ---- Deteksi (YOLO atau Statis) ----
        boxes_data = []
        if USE_YOLO:
            yolo_results = model(
                frame,
                conf=YOLO_CONF,
                device=YOLO_DEVICE if USE_GPU else "cpu",
                verbose=False,
                imgsz=640
            )
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_yolo = float(box.conf[0])
                    cls_id    = int(box.cls[0])
                    cls_name  = model.names[cls_id]
                    boxes_data.append((x1, y1, x2, y2, conf_yolo, cls_name))
        else:
            fh, fw = frame.shape[:2]
            box_w, box_h = 300, 150
            x1, y1 = max(0, (fw - box_w) // 2), max(0, (fh - box_h) // 2)
            x2, y2 = min(fw, x1 + box_w), min(fh, y1 + box_h)
            boxes_data.append((x1, y1, x2, y2, 1.0, "Center ROI"))

        debug_crops = []
        new_cache   = {}

        for (x1, y1, x2, y2, conf_yolo, cls_name) in boxes_data:

                # Crop dengan padding
                pad = CROP_PAD
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(frame.shape[1], x2 + pad)
                cy2 = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                # ---- OCR (setiap N frame) ----
                box_id = f"{x1}_{y1}_{x2}_{y2}"

                if frame_count % OCR_SKIP_FRAMES == 0:
                    ocr_hits = run_ocr(crop)
                    if ocr_hits:
                        ocr_text = " / ".join(
                            [f"{h['text']}({h['confidence']:.0%})"
                             for h in ocr_hits]
                        )
                        new_cache[box_id] = ocr_text
                    else:
                        new_cache[box_id] = ocr_cache.get(box_id, "?")
                else:
                    new_cache[box_id] = ocr_cache.get(box_id, "...")

                ocr_text = new_cache[box_id]

                # ---- Warna berdasarkan class ----
                color_map = {
                    "1": (0, 255, 255),
                    "2": (0, 200, 255),
                    "3": (0, 255, 100),
                    "4": (100, 100, 255),
                    "5": (255, 100, 200),
                }
                color = color_map.get(cls_name, (0, 255, 0))

                # ---- Gambar bounding box ----
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Label background
                label_yolo = f"{cls_name} ({conf_yolo:.0%})"
                label_ocr  = f"OCR: {ocr_text}"

                (lw, lh), _ = cv2.getTextSize(
                    label_yolo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(output,
                              (x1, y1 - lh - 20), (x1 + lw + 6, y1),
                              color, -1)
                cv2.putText(output, label_yolo,
                            (x1 + 3, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(output, label_ocr,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2, cv2.LINE_AA)

                # Simpan untuk debug
                debug_crops.append((crop, cls_name, ocr_text))

        # Update cache
        ocr_cache = new_cache

        # ---- Overlay Info ----
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (280, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

        cv2.putText(output, f"FPS: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"Deteksi: {len(debug_crops)} objek",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(output, f"Model: {MODEL_PATH.split('.')[0]}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # ---- Tampilkan ----
        cv2.imshow("YOLO + OCR | Bearing Number Reader", output)

        # ---- Debug Window ----
        if show_debug and debug_crops:
            debug_imgs = []
            for crop_img, name, ocr in debug_crops[:4]:  # max 4
                processed, enhanced = preprocess_for_metal_ocr(crop_img)
                # Tampilkan 3 versi berdampingan
                h_target = 150
                w_orig   = int(crop_img.shape[1] * h_target / max(crop_img.shape[0], 1))

                orig_r  = cv2.resize(crop_img, (w_orig, h_target))
                enh_r   = cv2.resize(
                    cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                    (w_orig, h_target))
                proc_r  = cv2.resize(
                    cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
                    (w_orig, h_target))

                row = np.hstack([orig_r, enh_r, proc_r])
                cv2.putText(row, f"{name}: {ocr}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                debug_imgs.append(row)

            max_w = max(img.shape[1] for img in debug_imgs)
            padded = [np.hstack([img,
                       np.zeros((img.shape[0], max_w - img.shape[1], 3),
                                dtype=np.uint8)])
                      for img in debug_imgs]
            debug_panel = np.vstack(padded)
            cv2.putText(debug_panel, "Orig | CLAHE | Processed",
                        (5, debug_panel.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Debug: Pre-Processing", debug_panel)

        # ---- Keyboard ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Keluar...")
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"ocr_screenshot_{ts}.jpg"
            cv2.imwrite(fname, output)
            print(f"[INFO] Screenshot disimpan: {fname}")
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Debug: Pre-Processing")
            print(f"[INFO] Debug view: {'ON' if show_debug else 'OFF'}")
        elif key == ord('r'):
            ocr_cache.clear()
            print("[INFO] OCR cache direset")

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline selesai.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  YOLO + OCR | Bearing Number Reader v3.0")
    print("=" * 50)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Kamera : {CAMERA_SOURCE}")
    print(f"  GPU    : {'Ya (CUDA)' if USE_GPU else 'Tidak (CPU)'}")
    print("=" * 50)

    run_realtime()