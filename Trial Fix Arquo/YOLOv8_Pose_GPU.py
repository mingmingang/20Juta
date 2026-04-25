import cv2
import time
import os
import shutil
import torch
from ultralytics import YOLO

# [OPTIMASI CPU] Maksimalkan penggunaan thread CPU (Ryzen) untuk video decoding OpenCV
cv2.setNumThreads(cv2.getNumberOfCPUs())


# ============================================================
# AUTO-DETECT DEVICE: CUDA (NVIDIA) → DirectML (AMD iGPU) → CPU
# ============================================================
MODEL_NAME = "yolov8n-pose"  # ganti ke yolov8s-pose untuk lebih akurat
ONNX_PATH = f"{MODEL_NAME}.onnx"
INFER_WIDTH = 640  # Lebar frame yang dikirim ke iGPU — lebih kecil = transfer data lebih ringan

if torch.cuda.is_available():
    # NVIDIA GPU via CUDA
    DEVICE = 'cuda'
    model = YOLO(f"{MODEL_NAME}.pt")
    print(f"[INFO] Menggunakan: NVIDIA GPU (CUDA)")

else:
    # Cek apakah onnxruntime-directml tersedia (AMD Ryzen iGPU)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()

        if 'DmlExecutionProvider' in providers:
            # AMD iGPU via DirectML
            DEVICE = 'dml'

            # Export ke ONNX versi khusus AMD: FP16 (Setengah Presisi)
            ONNX_PATH = f"{MODEL_NAME}_amd_opt.onnx"
            
            if not os.path.exists(ONNX_PATH):
                print("[INFO] Exporting model ke ONNX (Versi Optimasi FP16 + Simplify)...")
                print("[INFO] Proses ini memakan waktu sebentar, tapi akan membuat AMD iGPU jauh lebih kencang!")
                temp_model = YOLO(f"{MODEL_NAME}.pt")
                # half=True: Perhitungan komputasi diringankan ke 16-bit (Lebih hemat memory bandwidth iGPU)
                # simplify=True: Menyederhanakan grafis ONNX agar CPU & iGPU tidak kerja 2x lipat memproses layer kosong
                exported_file = temp_model.export(format="onnx", imgsz=640, half=True, simplify=True)
                
                # Copy hasil export ke file khusus
                if os.path.exists(exported_file):
                    shutil.move(exported_file, ONNX_PATH)
                print("[INFO] Export selesai!")

            model = YOLO(ONNX_PATH, task='pose')
            print(f"[INFO] Menggunakan: AMD Ryzen iGPU (DirectML) - FULL OPTIMIZED")
        else:
            raise ImportError("DirectML tidak tersedia")

    except ImportError:
        # Fallback ke CPU
        DEVICE = 'cpu'
        model = YOLO(f"{MODEL_NAME}.pt")
        print(f"[INFO] Menggunakan: CPU")
        print(f"[TIP] Install DirectML untuk pakai AMD iGPU: pip install onnxruntime-directml")

# Sumber video — ganti sesuai kebutuhan
cap = cv2.VideoCapture(1)  # Kamera
#cap = cv2.VideoCapture(r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\QC Cylinder Block CCTV.MOV")

# Deteksi otomatis: kamera atau file
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
USE_CAMERA = total_frames <= 0

# FPS video untuk sinkronisasi (hanya untuk file)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30

start_time = time.time()
frame_count = 0
prev_fps_time = start_time

while cap.isOpened():
    if USE_CAMERA:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        # File video: sinkronisasi real-time dengan skip frame
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

    # [OPTIMASI] Pre-resize sebelum inference — kurangi data transfer CPU → iGPU
    h, w = frame.shape[:2]
    if w > INFER_WIDTH:
        scale = INFER_WIDTH / w
        infer_frame = cv2.resize(frame, (INFER_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        infer_frame = frame

    # YOLOv8 Pose detection
    if DEVICE == 'dml':
        results = model(infer_frame, verbose=False)  # DirectML via ONNX Runtime
    else:
        results = model(infer_frame, device=DEVICE, verbose=False)

    # Gambar hasil deteksi (pose keypoints + bounding box)
    annotated = results[0].plot()

    # Hitung & tampilkan FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_fps_time) if (curr_time - prev_fps_time) > 0 else 0
    prev_fps_time = curr_time
    cv2.putText(annotated, f'FPS: {int(fps)} | {DEVICE.upper()}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Pose (GPU)', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
