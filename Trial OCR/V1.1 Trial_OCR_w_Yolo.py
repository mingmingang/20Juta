import cv2
import os
import numpy as np
from ultralytics import YOLO
from easyocr import Reader

# ============================================================
#   OCR BEARING NUMBER - Real-Time Kamera (YOLO + EasyOCR)
# ============================================================

def preprocess_bearing(image):
    """
    Preprocessing ringan untuk angka di atas logam melengkung (bearing).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 3
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(big)
    blur = cv2.medianBlur(enhanced, 3)
    return blur

def main():
    # 1. Tentukan path file model
    MODEL_PATH = "best_model_deteksi_roi_v2.pt" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model YOLO '{MODEL_PATH}' tidak ditemukan di folder ini!")
        print("Silakan copy/pindahkan file .pt hasil training Anda ke folder ini.")
        return

    # 2. Inisialisasi Model
    print("[INFO] Memuat model YOLO...")
    yolo_model = YOLO(MODEL_PATH)
    
    print("[INFO] Memuat EasyOCR...")
    reader = Reader(['en'], gpu=True) 

    # 3. Buka Kamera
    # Jika aplikasi IP Camera Lite membutuhkan username dan password, format URL-nya adalah: 
    # "http://username:password@IP:PORT/video"
    # Jika ingin kembali menggunakan webcam bawaan/default, ubah nilainya menjadi: CAMERA_SOURCE = 0
    CAMERA_SOURCE = "http://admin:admin@10.1.24.123:8081/video"
    
    print(f"[INFO] Membuka kamera dari sumber: {CAMERA_SOURCE} ...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera. Pastikan terhubung dengan baik.")
        return

    print("\n[INFO] Kamera berhasil dibuka!")
    print("[INFO] Dekatkan bearing ke kamera. Tekan tombol 'q' pada keyboard untuk keluar.\n")

    # 4. Looping Video Secara Real-Time
    while True:
        # Baca satu frame dari kamera
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal menangkap gambar dari kamera.")
            break
            
        vis_img = frame.copy() # Gambar untuk visualisasi kotak & teks
        
        # Eksekusi YOLO pada frame (verbose=False agar terminal tidak terlalu penuh)
        results = yolo_model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Ambil nama kelas
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                
                # Jika yang terdeteksi adalah bearing
                if class_name == 'number_bearing':
                    # Koordinat Kotak
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Validasi batas agar crop tidak error jika kotak keluar dari layar kamera
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Jika area ukurannya aneh, lewati
                    if y2 <= y1 or x2 <= x1:
                        continue
                        
                    # Crop & Preprocessing
                    crop_img = frame[y1:y2, x1:x2]
                    processed_crop = preprocess_bearing(crop_img)
                    
                    # OCR
                    ocr_results = reader.readtext(processed_crop, allowlist='0123456789')
                    
                    # Menulis hasil ke layar
                    if ocr_results:
                        best_text = ocr_results[0][1]
                        conf = ocr_results[0][2]
                        
                        label = f"No: {best_text} ({conf:.2f})"
                        color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                        
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(vis_img, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(vis_img, "Mencari...", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Tampilkan Jendela Video
        cv2.imshow("Real-Time Bearing OCR", vis_img)
        
        # Logika Keluar: Tekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Menutup program...")
            break

    # 5. Bersihkan Memori
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
