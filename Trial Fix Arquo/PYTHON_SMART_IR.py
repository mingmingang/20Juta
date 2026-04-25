import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ==================== Preprocessing Config ====================
PROCESS_WIDTH = 640           # Lebar frame untuk inferensi (lebih kecil = lebih cepat)
PROCESS_HEIGHT = 480          # Tinggi frame untuk inferensi
SKIP_FRAMES = 2               # Proses setiap N frame (1 = semua, 2 = skip 1, 3 = skip 2, dst.)
BLUR_KERNEL = (3, 3)          # Kernel GaussianBlur untuk noise reduction (set None untuk skip)
YOLO_CONF = 0.5               # Confidence threshold YOLO
YOLO_IMG_SIZE = 320           # Imgsz YOLO inference (320 = ringan, 640 = default)

def preprocess_frame(frame, target_width=PROCESS_WIDTH, target_height=PROCESS_HEIGHT, blur_kernel=BLUR_KERNEL):
    """Preprocessing frame: resize + optional blur untuk meringankan inferensi."""
    # Resize ke resolusi lebih kecil
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    # Optional: Gaussian Blur untuk kurangi noise (membantu model fokus ke fitur penting)
    if blur_kernel is not None:
        resized = cv2.GaussianBlur(resized, blur_kernel, 0)
    return resized

# ==================== YOLO Setup (Kamera 1 - Custom Model) ====================
model = YOLO(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\First Trial (Snitching 3 CV)\last (8).pt")
print("Class Names (Custom):", model.names)

# ==================== YOLO Setup (Kamera 2 - Pretrained COCO All Classes) ====================
model_coco = YOLO("yolo11n.pt")  # Model pretrained COCO YOLO11 Nano (otomatis download jika belum ada)
print("Class Names (COCO):", model_coco.names)

# ==================== Mediapipe Hand Landmarker Setup (Tasks API) ====================
MODEL_PATH = os.path.join(
    r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\First Trial (Snitching 3 CV)",
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)
hand_landmarker = HandLandmarker.create_from_options(options)

# Indeks landmark ujung jari (MediaPipe Hand Landmark)
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Hand connections untuk menggambar skeleton tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                 # Palm
]


# Fungsi untuk mengenali gesture tangan
def recognize_gesture(landmarks):
    """Mengenali gesture berdasarkan posisi ujung jari (normalized landmarks)."""
    ujung_jempol = landmarks[THUMB_TIP]
    ujung_telunjuk = landmarks[INDEX_TIP]
    ujung_jariTengah = landmarks[MIDDLE_TIP]
    ujung_jarimanis = landmarks[RING_TIP]
    ujung_kelingking = landmarks[PINKY_TIP]

    # Thumbs Up => jika hanya jempol yang diangkat
    if (ujung_jempol.y < ujung_telunjuk.y and
        ujung_jempol.y < ujung_jariTengah.y and
        ujung_jempol.y < ujung_jarimanis.y and
        ujung_jempol.y < ujung_kelingking.y):
        return "Thumbs Up"

    # Peace Sign (Index and Middle finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y and
        ujung_kelingking.y > ujung_jempol.y):
        return "Peace Sign"

    # Fist
    if (ujung_jempol.y > ujung_telunjuk.y and
        ujung_jempol.y > ujung_jariTengah.y and
        ujung_jempol.y > ujung_jarimanis.y and
        ujung_jempol.y > ujung_kelingking.y):
        return "Fist"

    # Metal Sign (Index and Pinky finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_kelingking.y < ujung_jempol.y and
        ujung_jariTengah.y > ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y):
        return "Metal"

    return "Gesture tidak diketahui"


def draw_hand_landmarks(image, landmarks, connections, h, w):
    """Menggambar landmark dan koneksi tangan pada image menggunakan OpenCV."""
    # Gambar koneksi (garis)
    for start_idx, end_idx in connections:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Gambar titik landmark
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


def detect_hand_gesture(image, landmarker):
    """Mendeteksi tangan dan gesture menggunakan Mediapipe Tasks API."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = landmarker.detect(mp_image)
    gesture_text = ""
    h, w, _ = image.shape

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            # Deteksi gesture
            gesture_text = recognize_gesture(hand_lms)
            # Gambar landmarks pada frame
            draw_hand_landmarks(image, hand_lms, HAND_CONNECTIONS, h, w)

    return image, gesture_text


# ==================== Buka Kamera 1 (Webcam - YOLO Custom + Mediapipe) ====================
cap1 = cv2.VideoCapture(1)
# Set resolusi capture lebih rendah agar kamera tidak kirim frame terlalu besar
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer agar tidak lag

if not cap1.isOpened():
    print("Tidak dapat membuka kamera 1 (webcam)")
    exit()

# ==================== Buka Kamera 2 (IP Camera - YOLO Mouse Detection) ====================
cap2 = cv2.VideoCapture("http://10.120.244.45:4747/video")
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap2.isOpened():
    print("Tidak dapat membuka kamera 2 (IP camera). Pastikan DroidCam aktif.")
    exit()

print("Kedua kamera berhasil dibuka. Tekan 'q' untuk keluar.")
print(f"Preprocessing: resize={PROCESS_WIDTH}x{PROCESS_HEIGHT}, skip_frames={SKIP_FRAMES}, blur={BLUR_KERNEL}, yolo_imgsz={YOLO_IMG_SIZE}")

# FPS tracking variables
prev_time1 = 0
prev_time2 = 0
frame_count = 0  # Counter untuk frame skipping

# Cache hasil deteksi terakhir (untuk frame yang di-skip)
last_annotated1 = None
last_annotated2 = None
last_gesture = ""

while True:
    frame_count += 1
    should_process = (frame_count % SKIP_FRAMES == 0)  # Proses hanya setiap N frame

    # ========== Kamera 1: YOLO Custom + Mediapipe Hand Gesture ==========
    curr_time1 = time.time()
    ret1, frame1 = cap1.read()
    if not ret1:
        print("Gagal menangkap frame dari kamera 1")
        break

    if should_process:
        # Preprocessing: resize + blur
        processed1 = preprocess_frame(frame1)

        # YOLO Detection (Custom Model) dengan imgsz kecil + conf threshold
        results1 = model(processed1, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)
        boxes1 = results1[0].boxes.data

        person_count = 0
        helmet_count = 0
        no_helmet_count = 0
        glasses_count = 0
        no_glasses_count = 0

        for box in boxes1:
            cls_id = int(box[-1])
            class_name = model.names[cls_id].lower().replace(" ", "")
            if class_name == "people":
                person_count += 1
            elif class_name == "helmet":
                helmet_count += 1
            elif class_name == "nohelmet":
                no_helmet_count += 1
            elif class_name == "glasses":
                glasses_count += 1
            elif class_name == "noglasses":
                no_glasses_count += 1

        annotated_frame1 = results1[0].plot()

        # Mediapipe Hand Gesture Detection (pakai frame yang sudah di-resize)
        annotated_frame1, gesture = detect_hand_gesture(annotated_frame1, hand_landmarker)
        last_gesture = gesture

        # Tampilkan Info Kamera 1
        cv2.putText(annotated_frame1, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame1, f"Helmets: {helmet_count}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame1, f"No Helmets: {no_helmet_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(annotated_frame1, f"Glasses: {glasses_count}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame1, f"No Glasses: {no_glasses_count}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if last_gesture:
            cv2.putText(annotated_frame1, f"Hand Gesture: {last_gesture}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        last_annotated1 = annotated_frame1.copy()

    # Tampilkan frame terakhir yang di-proses (atau cache jika frame di-skip)
    if last_annotated1 is not None:
        display1 = last_annotated1
    else:
        display1 = preprocess_frame(frame1)

    # Hitung dan tampilkan FPS Kamera 1
    fps1 = 1 / (curr_time1 - prev_time1) if prev_time1 > 0 else 0
    prev_time1 = curr_time1
    cv2.putText(display1, f"FPS: {fps1:.1f}", (display1.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Kamera 1 - YOLO Custom + Hand Gesture", display1)

    # ========== Kamera 2: YOLO Pretrained (All COCO Classes) ==========
    curr_time2 = time.time()
    ret2, frame2 = cap2.read()
    if not ret2:
        print("Gagal menangkap frame dari kamera 2")
        break

    if should_process:
        # Preprocessing: resize + blur
        processed2 = preprocess_frame(frame2)

        # YOLO Detection (Pretrained YOLO Nano) dengan imgsz kecil
        results2 = model_coco(processed2, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)
        annotated_frame2 = results2[0].plot()

        obj_count = len(results2[0].boxes.data)

        # Tampilkan Info Kamera 2
        cv2.putText(annotated_frame2, f"Objects Detected: {obj_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        last_annotated2 = annotated_frame2.copy()

    # Tampilkan frame terakhir yang di-proses (atau cache jika frame di-skip)
    if last_annotated2 is not None:
        display2 = last_annotated2
    else:
        display2 = preprocess_frame(frame2)

    # Hitung dan tampilkan FPS Kamera 2
    fps2 = 1 / (curr_time2 - prev_time2) if prev_time2 > 0 else 0
    prev_time2 = curr_time2
    cv2.putText(display2, f"FPS: {fps2:.1f}", (display2.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Kamera 2 - YOLO Nano Object Detection (COCO)", display2)

    # Tekan 'q' untuk keluar dari kedua kamera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
hand_landmarker.close()
cv2.destroyAllWindows()