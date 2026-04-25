import cv2
import mediapipe as mp
import urllib.request
import os
import time
import numpy as np
from multiprocessing import Process, Queue, Value
import ctypes

# Mengaktifkan Akselerasi GPU (OpenCL) untuk Preprocessing OpenCV
cv2.ocl.setUseOpenCL(True)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Memulai Inisialisasi Program...")

# ==========================================
# 1. DOWNLOAD MODEL
# ==========================================
det_model_path = 'efficientdet_lite0.tflite'
pose_model_path = 'pose_landmarker_lite.task'
hand_model_path = 'hand_landmarker.task'

if not os.path.exists(det_model_path):
    print("Mengunduh model Object Detection...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite', det_model_path)
if not os.path.exists(pose_model_path):
    print("Mengunduh model Pose Landmarker...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', pose_model_path)
if not os.path.exists(hand_model_path):
    print("Mengunduh model Hand Landmarker...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', hand_model_path)
    
print("Model siap!")

# ==========================================
# KONEKSI SKELETON & TANGAN
# ==========================================
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# ==========================================
# 2. AI WORKER (PROSES TERPISAH)
# ==========================================
def ai_process_worker(input_queue, output_queue, running_flag):
    """Proses terpisah: menjalankan semua AI di CPU core lain."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    
    det = mp_vision.ObjectDetector.create_from_options(mp_vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
        score_threshold=0.5
    ))
    pose = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
        output_segmentation_masks=False,
    ))
    hand = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=2
    ))
    
    print("[AI Process] Siap di CPU core terpisah.")
    frame_counter = 0
    
    while running_flag.value:
        try:
            frame_rgb = input_queue.get(timeout=0.1)
        except:
            continue
        
        frame_counter += 1
        h_orig, w_orig = frame_rgb.shape[:2]
        
        # Resize ke 320x240 untuk AI (jauh lebih ringan!)
        small = cv2.resize(frame_rgb, (320, 240))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        h_s, w_s = 240, 320
        
        # Jalankan Pose & Hand
        p_res = pose.detect(mp_image)
        h_res = hand.detect(mp_image)
        
        # Object Detection tiap 3 frame saja
        d_data = []
        if frame_counter % 3 == 1:
            d_res = det.detect(mp_image)
            for detection in d_res.detections:
                bbox = detection.bounding_box
                d_data.append({
                    'name': detection.categories[0].category_name,
                    'score': round(detection.categories[0].score, 2),
                    'x': int(bbox.origin_x * w_orig / w_s),
                    'y': int(bbox.origin_y * h_orig / h_s),
                    'w': int(bbox.width * w_orig / w_s),
                    'h': int(bbox.height * h_orig / h_s),
                })
        
        # Konversi ke angka biasa (bisa lewat Queue)
        pose_pts = []
        if p_res.pose_landmarks:
            for lms in p_res.pose_landmarks:
                pose_pts.append([(lm.x * w_orig, lm.y * h_orig) for lm in lms])
        
        hand_pts = []
        if h_res.hand_landmarks:
            for lms in h_res.hand_landmarks:
                hand_pts.append([(lm.x * w_orig, lm.y * h_orig) for lm in lms])
        
        # Kosongkan antrian lama, kirim yang terbaru
        while not output_queue.empty():
            try: output_queue.get_nowait()
            except: break
        
        output_queue.put({'pose': pose_pts, 'hand': hand_pts, 'det': d_data})
    
    det.close(); pose.close(); hand.close()
    print("[AI Process] Berhenti.")


# ==========================================
# 3. FUNGSI RENDER OVERLAY (Cached)
# ==========================================
def render_cached_overlay(h, w, pose_data, hand_data):
    """Render skeleton & tangan ke gambar overlay hitam transparan.
    Ini hanya dipanggil saat ada data AI BARU, bukan setiap frame."""
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    for pts in pose_data:
        int_pts = [(int(x), int(y)) for x, y in pts]
        for s, e in POSE_CONNECTIONS:
            if s < len(int_pts) and e < len(int_pts):
                p1, p2 = int_pts[s], int_pts[e]
                if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                    cv2.line(overlay, p1, p2, (0, 255, 255), 2)
        for cx, cy in int_pts:
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
    
    for pts in hand_data:
        int_pts = [(int(x), int(y)) for x, y in pts]
        for s, e in HAND_CONNECTIONS:
            if s < len(int_pts) and e < len(int_pts):
                p1, p2 = int_pts[s], int_pts[e]
                if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                    cv2.line(overlay, p1, p2, (255, 105, 180), 2)
        for cx, cy in int_pts:
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)
    
    # Pre-compute mask juga agar main loop tidak perlu menghitung ulang!
    mask = np.any(overlay > 0, axis=2)
    return overlay, mask


# ==========================================
# 4. MAIN PROCESS
# ==========================================
if __name__ == '__main__':
    input_q = Queue(maxsize=2)
    output_q = Queue(maxsize=2)
    running_flag = Value(ctypes.c_bool, True)
    
    ai_proc = Process(target=ai_process_worker, args=(input_q, output_q, running_flag), daemon=True)
    ai_proc.start()
    
    # OPTIMASI: Gunakan MOSSE tracker (10x lebih cepat dari KCF!)
    # KCF sangat lambat pada bounding box besar (orang = box besar)
    try:
        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker_type = "MOSSE"
    except AttributeError:
        try:
            tracker = cv2.TrackerKCF_create()
            tracker_type = "KCF"
        except AttributeError:
            tracker = cv2.TrackerMIL_create()
            tracker_type = "MIL"
    
    print(f"Tracker: {tracker_type}")
    
    tracking_active = False
    tracked_category = ""
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nHalo! Menjalankan program...\nTekan 'q' untuk keluar.\n")
    
    prev_time = 0
    cached_overlay = None     # Gambar overlay yang sudah di-render
    cached_mask = None        # Mask overlay (pre-computed)
    current_det = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fh, fw = frame.shape[:2]
        
        # GPU Preprocessing
        umat = cv2.UMat(frame)
        umat_rgb = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
        frame_rgb = umat_rgb.get()
        
        # Kirim frame ke AI Process (non-blocking)
        if input_q.empty():
            try: input_q.put_nowait(frame_rgb)
            except: pass
        
        # Cek apakah ada hasil AI baru
        new_ai = False
        try:
            result = output_q.get_nowait()
            new_ai = True
            if result['det']:
                current_det = result['det']
            
            # OPTIMASI KUNCI: Hanya render overlay saat ada DATA BARU dari AI!
            # Pada frame biasa (tanpa data baru), kita pakai overlay yang sudah di-cache.
            # Ini berarti menggambar 150+ circle/line hanya terjadi ~5-10x/detik, 
            # BUKAN 30x/detik!
            cached_overlay, cached_mask = render_cached_overlay(fh, fw, result['pose'], result['hand'])
        except:
            pass
        
        # Tempel overlay yang sudah di-cache (SANGAT CEPAT - hanya numpy indexing!)
        if cached_overlay is not None and cached_mask is not None:
            frame[cached_mask] = cached_overlay[cached_mask]
        
        # ------------------------------------------
        # OBJECT TRACKING (MOSSE = super ringan)
        # ------------------------------------------
        if not tracking_active:
            if len(current_det) > 0:
                target_found = False
                for det in current_det:
                    if det['name'] == 'person':
                        init_bbox = (det['x'], det['y'], det['w'], det['h'])
                        try:
                            tracker = cv2.legacy.TrackerMOSSE_create()
                        except:
                            tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, init_bbox)
                        tracking_active = True
                        tracked_category = det['name']
                        target_found = True
                        break
                
                if not target_found:
                    det = current_det[0]
                    init_bbox = (det['x'], det['y'], det['w'], det['h'])
                    try:
                        tracker = cv2.legacy.TrackerMOSSE_create()
                    except:
                        tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, init_bbox)
                    tracking_active = True
                    tracked_category = det['name']
            else:
                cv2.putText(frame, 'Mencari Objek / Orang...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'TRACKING: {tracked_category}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Objek Hilang! Reset...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                try:
                    tracker = cv2.legacy.TrackerMOSSE_create()
                except:
                    tracker = cv2.TrackerKCF_create()
                tracking_active = False
        
        # FPS Counter (warna dinamis)
        now = time.time()
        fps = 1 / (now - prev_time) if prev_time != 0 else 0
        prev_time = now
        color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f'FPS: {int(fps)}', (fw - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Kamera - Pose & Tracking V3', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    running_flag.value = False
    ai_proc.join(timeout=3)
    if ai_proc.is_alive():
        ai_proc.terminate()
    cap.release()
    cv2.destroyAllWindows()
