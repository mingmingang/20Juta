import mediapipe as mp
import cv2
import time
import signal
import sys
import numpy as np
import urllib.request
import os

# ============================================================
# OPTIMASI: Bagi tugas antara AMD iGPU (OpenCL) dan CPU (Ryzen)
# ============================================================

cv2.setNumThreads(cv2.getNumberOfCPUs())

ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)
if ocl_available:
    dev = cv2.ocl.Device.getDefault()
    print(f"[INFO] OpenCL aktif → {dev.name()} (iGPU AMD)")
    print(f"[INFO] Pre-processing (resize + cvtColor) akan dikerjakan iGPU")
else:
    print("[INFO] OpenCL tidak tersedia, semua dikerjakan CPU")
print(f"[INFO] Mediapipe inference → CPU ({cv2.getNumberOfCPUs()} thread)")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# === FITUR OPSIONAL: Object Detection (uncomment untuk aktifkan) ===
ENABLE_OBJECT_DETECTION = False  # Ganti ke True untuk aktifkan

# # Download model EfficientDet jika belum ada
# det_model_path = 'efficientdet_lite0.tflite'
# if not os.path.exists(det_model_path):
#     print("Mengunduh model Object Detection...")
#     urllib.request.urlretrieve(
#         'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
#         det_model_path)
#     print("Model Object Detection siap!")

if ENABLE_OBJECT_DETECTION:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    det_model_path = 'efficientdet_lite0.tflite'
    if not os.path.exists(det_model_path):
        print("Mengunduh model Object Detection...")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
            det_model_path)
    obj_detector = mp_vision.ObjectDetector.create_from_options(
        mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=det_model_path),
            score_threshold=0.5
        ))
    print("[INFO] Object Detection: AKTIF (EfficientDet Lite0, 90 kelas COCO)")
    det_frame_counter = 0
    current_detections = []
else:
    print("[INFO] Object Detection: NONAKTIF (set ENABLE_OBJECT_DETECTION = True untuk aktifkan)")

# === PILIH SUMBER VIDEO (uncomment salah satu) ===
cap = cv2.VideoCapture(1)  # Kamera langsung
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\QC Cylinder Block HP.MOV")
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\QC Cylinder Block CCTV.MOV")
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\Video Testing Dance.mp4")

# Lebar frame untuk processing Mediapipe (lebih kecil = lebih cepat)
PROCESS_WIDTH = 640

# Deteksi otomatis: kamera atau file
USE_CAMERA = isinstance(cap.get(cv2.CAP_PROP_FRAME_COUNT), float) and cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0

# FPS asli video untuk sinkronisasi (hanya untuk file)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30

# ==========================================
# FUNGSI: Validasi & Bounding Box dari Pose
# ==========================================
def is_valid_pose(pose_landmarks, min_visible=10, min_visibility=0.65):
    """Cek apakah pose yang terdeteksi benar-benar manusia."""
    if pose_landmarks is None:
        return False
    
    lms = pose_landmarks.landmark
    
    # DEBUG: print visibility dari landmark kunci
    print(f"[DEBUG POSE] Bahu L:{lms[11].visibility:.2f} R:{lms[12].visibility:.2f} | "
          f"Pinggul L:{lms[23].visibility:.2f} R:{lms[24].visibility:.2f} | "
          f"Lutut L:{lms[25].visibility:.2f} R:{lms[26].visibility:.2f}")
    
    # WAJIB: kedua bahu (11,12) dan kedua pinggul (23,24) harus terlihat
    KEY_LANDMARKS = [11, 12, 23, 24]
    for idx in KEY_LANDMARKS:
        if lms[idx].visibility < min_visibility:
            print(f"[DEBUG POSE] DITOLAK: landmark {idx} visibility {lms[idx].visibility:.2f} < {min_visibility}")
            return False
    
    # Cek jumlah landmark visible
    visible_count = sum(1 for lm in lms if lm.visibility >= min_visibility)
    if visible_count < min_visible:
        print(f"[DEBUG POSE] DITOLAK: hanya {visible_count} visible (min={min_visible})")
        return False
    
    # Cek proporsi tubuh
    xs = [lm.x for lm in lms if lm.visibility >= min_visibility]
    ys = [lm.y for lm in lms if lm.visibility >= min_visibility]
    
    if len(xs) < 2:
        return False
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    ratio = height / width if width > 0 else 0
    print(f"[DEBUG POSE] Proporsi h/w: {ratio:.2f} (min 0.5)")
    
    if height < width * 0.5:
        print(f"[DEBUG POSE] DITOLAK: terlalu lebar")
        return False
    
    print(f"[DEBUG POSE] DITERIMA ✓ ({visible_count} visible, rasio {ratio:.2f})")
    return True

def get_pose_bbox(pose_landmarks, w, h, padding=30):
    """Buat bounding box dari pose landmarks dengan padding."""
    if not is_valid_pose(pose_landmarks):
        return None
    
    xs = [lm.x * w for lm in pose_landmarks.landmark]
    ys = [lm.y * h for lm in pose_landmarks.landmark]
    
    x_min = max(0, int(min(xs)) - padding)
    y_min = max(0, int(min(ys)) - padding)
    x_max = min(w, int(max(xs)) + padding)
    y_max = min(h, int(max(ys)) + padding)
    
    bw = x_max - x_min
    bh = y_max - y_min
    
    if bw < 10 or bh < 10:
        return None
    
    return (x_min, y_min, bw, bh)

# ==========================================
# SIGNAL HANDLER untuk Ctrl+C
# ==========================================
def signal_handler(sig, frame):
    print("\n[Ctrl+C] Menghentikan program...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# TRACKER SETUP
# ==========================================
def create_tracker():
    """Buat tracker MOSSE baru (atau fallback ke KCF/MIL)."""
    try:
        return cv2.legacy.TrackerMOSSE_create(), "MOSSE"
    except AttributeError:
        try:
            return cv2.TrackerKCF_create(), "KCF"
        except AttributeError:
            return cv2.TrackerMIL_create(), "MIL"

tracker, tracker_type = create_tracker()
tracking_active = False
print(f"[INFO] Tracker: {tracker_type}")
print("\nHalo! Menjalankan program...\nTekan 'q' atau ESC untuk keluar.\n")

# ==========================================
# MAIN LOOP
# ==========================================
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    start_time = time.time()
    frame_count = 0
    prev_fps_time = start_time

    while cap.isOpened():
        # --- Frame Capture (dengan frame-skip untuk video file) ---
        if USE_CAMERA:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            elapsed = time.time() - start_time
            target_frame = int(elapsed * video_fps)

            skipped = False
            while frame_count < target_frame:
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1
                skipped = True

            if not skipped:
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1

            if not ret:
                break

            ret, frame = cap.retrieve()
            if not ret:
                break

        h, w = frame.shape[:2]

        # --- [AMD iGPU via OpenCL] Pre-processing ---
        umat_frame = cv2.UMat(frame)
        scale = PROCESS_WIDTH / w
        small_umat = cv2.resize(umat_frame, (PROCESS_WIDTH, int(h * scale)))
        small_rgb_umat = cv2.cvtColor(small_umat, cv2.COLOR_BGR2RGB)
        small_rgb = small_rgb_umat.get()

        # --- [CPU] Mediapipe Holistic inference ---
        small_rgb.flags.writeable = False
        results = holistic.process(small_rgb)

        # --- Gambar landmark di frame original ---
        image = frame.copy()
        pose_valid = is_valid_pose(results.pose_landmarks)

        # Hanya gambar skeleton & hands jika pose VALID (benar-benar manusia)
        if pose_valid:
            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

            # Left hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

            # Pose skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # --- OBJECT DETECTION (jika diaktifkan) ---
        if ENABLE_OBJECT_DETECTION:
            det_frame_counter += 1
            # Jalankan deteksi tiap 3 frame agar tidak terlalu berat
            if det_frame_counter % 3 == 1:
                det_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_small = cv2.resize(det_rgb, (320, 240))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_small)
                det_results = obj_detector.detect(mp_img)
                h_s, w_s = 240, 320
                current_detections = []
                for detection in det_results.detections:
                    bbox = detection.bounding_box
                    current_detections.append({
                        'name': detection.categories[0].category_name,
                        'score': round(detection.categories[0].score, 2),
                        'x': int(bbox.origin_x * w / w_s),
                        'y': int(bbox.origin_y * h / h_s),
                        'w': int(bbox.width * w / w_s),
                        'h': int(bbox.height * h / h_s),
                    })
            # Gambar bounding box deteksi
            for det in current_detections:
                dx, dy, dw, dh = det['x'], det['y'], det['w'], det['h']
                cv2.rectangle(image, (dx, dy), (dx + dw, dy + dh), (255, 255, 0), 2)
                cv2.putText(image, f"{det['name']} {det['score']}", (dx, dy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --- POSE-BASED OBJECT TRACKING ---
        pose_bbox = get_pose_bbox(results.pose_landmarks, w, h)

        if pose_bbox is not None:
            # Pose terdeteksi → (re)init tracker dengan bbox dari skeleton
            tracker, _ = create_tracker()
            tracker.init(image, pose_bbox)
            tracking_active = True

            x, y, bw, bh = pose_bbox
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(image, 'TRACKING: person (pose)', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif tracking_active:
            # Pose hilang → fallback ke tracker
            success, box = tracker.update(image)
            if success:
                x, y, bw, bh = [int(v) for v in box]
                cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                cv2.putText(image, 'TRACKING: person (tracker)', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(image, 'Objek Hilang! Mencari...', (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tracking_active = False
        else:
            cv2.putText(image, 'Mencari Orang...', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # --- FPS Counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_fps_time) if (curr_time - prev_fps_time) > 0 else 0
        prev_fps_time = curr_time
        color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        label = "iGPU+CPU" if ocl_available else "CPU"
        cv2.putText(image, f'FPS: {int(fps)} | Src: {int(video_fps)}fps | {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Holistic + Tracking', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

print("\nMenghentikan program...")
cap.release()
cv2.destroyAllWindows()
print("Program berhenti.")
