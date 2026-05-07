import cv2
from ultralytics import YOLO

# model YOLOv11
model = YOLO("C:/Users/Victus/Downloads/best (20).pt")

# Sumber input: Gambar statis
sumber_gambar = r"C:\KULIAH\SEMESTER 6\T-MIND\05 Project TMMIN\Trial OCR\WhatsApp Image 2026-05-05 at 21.08.23 (2).jpeg"

frame = cv2.imread(sumber_gambar)

if frame is None:
    print(f"[ERROR] Gagal memuat gambar dari: {sumber_gambar}")
else:
    height, width = frame.shape[:2]
    print(f"[INFO] Resolusi Gambar: {width}x{height}")

    cv2.namedWindow('YOLOv11', cv2.WINDOW_NORMAL)
    # Uncomment baris di bawah ini jika ingin ukuran window awal persis mengikuti resolusi gambar
    # cv2.resizeWindow('YOLOv11', width, height)

    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLOv11', annotated_frame)
    print("[INFO] Tekan tombol apa saja pada keyboard untuk menutup gambar.")
    
    # Tunggu sampai pengguna menekan tombol apapun pada jendela gambar
    cv2.waitKey(0)

cv2.destroyAllWindows()