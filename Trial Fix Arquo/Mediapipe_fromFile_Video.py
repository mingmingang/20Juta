import mediapipe as mp
import cv2
import time

# ============================================================
# OPTIMASI: Bagi tugas antara AMD iGPU (OpenCL) dan CPU (Ryzen)
# ============================================================

# [CPU] Maksimalkan thread Ryzen untuk OpenCV dan Mediapipe
cv2.setNumThreads(cv2.getNumberOfCPUs())

# [AMD iGPU] Aktifkan OpenCL — OpenCV akan otomatis pakai Radeon Graphics
#            untuk operasi resize & cvtColor via UMat
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
mp_face_mesh = mp.solutions.face_mesh  # Untuk konstanta FACEMESH (mata, mulut)

#cap = cv2.VideoCapture(1)  # Kamera — uncomment untuk kembali pakai kamera
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\QC Cylinder Block HP.MOV")
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\QC Cylinder Block CCTV.MOV")
cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\Video Testing Dance.mp4")

# Lebar frame untuk processing Mediapipe (lebih kecil = lebih cepat)
PROCESS_WIDTH = 640

# Deteksi otomatis: kamera atau file
USE_CAMERA = isinstance(cap.get(cv2.CAP_PROP_FRAME_COUNT), float) and cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0

# Ambil FPS asli video untuk sinkronisasi (hanya untuk file)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    start_time = time.time()
    frame_count = 0
    prev_fps_time = start_time

    while cap.isOpened():
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

        # ── [AMD iGPU via OpenCL] Pre-processing ──────────────────────────
        # Konversi frame ke UMat agar resize & cvtColor jalan di Radeon iGPU
        umat_frame = cv2.UMat(frame)

        h, w = frame.shape[:2]
        scale = PROCESS_WIDTH / w
        small_umat = cv2.resize(umat_frame, (PROCESS_WIDTH, int(h * scale)))     # iGPU
        small_rgb_umat = cv2.cvtColor(small_umat, cv2.COLOR_BGR2RGB)              # iGPU
        small_rgb = small_rgb_umat.get()  # Kembalikan ke numpy untuk Mediapipe (CPU)
        # ──────────────────────────────────────────────────────────────────

        # ── [CPU] Mediapipe inference ──────────────────────────────────────
        small_rgb.flags.writeable = False
        results = holistic.process(small_rgb)
        # ──────────────────────────────────────────────────────────────────

        # Gambar landmark di frame ORIGINAL (koordinat normalized, tetap akurat)
        image = frame.copy()

        # 1. Face — FULL MESH (468 titik, berat — uncomment untuk aktifkan kembali)
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        # 1b. Face RINGAN — hanya mata kiri, mata kanan, dan mulut (uncomment untuk aktifkan)
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE,
        #                           mp_drawing.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=1))
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_RIGHT_EYE,
        #                           mp_drawing.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=1))
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_LIPS,
        #                           mp_drawing.DrawingSpec(color=(0,100,255), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(0,100,255), thickness=1, circle_radius=1))

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        # 3. Left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        # 4. Pose (termasuk titik mata, telinga, hidung, mulut di index 0-10)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Hitung & tampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_fps_time) if (curr_time - prev_fps_time) > 0 else 0
        prev_fps_time = curr_time
        label = "iGPU+CPU" if ocl_available else "CPU"
        cv2.putText(image, f'FPS: {int(fps)} | Cam: {int(video_fps)}fps | {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Mediapipe Holistic', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

