import cv2
from ultralytics import YOLO

# model YOLOv11
model = YOLO("C:/Users/Victus/Downloads/best (20).pt")

# Inisialisasi kamera
# Ganti IP_ADDRESS dan PORT sesuai dengan yang ada di aplikasi IP Camera Lite di HP Anda.
# Contoh URL: "http://admin:admin@10.1.24.123:8081/video"
#url_kamera = "http://admin:admin@10.1.24.123:8081/video"
url_kamera = "http://admin:admin@10.112.52.53:8081/video"
cap = cv2.VideoCapture(url_kamera)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Resolusi Kamera: {width}x{height}")

cv2.namedWindow('YOLOv11', cv2.WINDOW_NORMAL)
# Uncomment baris di bawah ini jika ingin ukuran window awal persis mengikuti resolusi kamera
# cv2.resizeWindow('YOLOv11', width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLOv11', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()