import cv2
import easyocr
import numpy as np

def preprocess_image(image_path, mode="BEARING"):
    """
    Melakukan preprocessing pada gambar untuk meningkatkan hasil OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar di path: {image_path}")

    # Mencegah Out-Of-Memory: Resize gambar jika ukurannya terlalu besar
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "BEARING":
        # Gunakan CLAHE yang lebih kuat untuk mengangkat kontras di area gelap (bearing kanan)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        processed_img = clahe.apply(gray)
    else:
        # Preprocessing khusus engine block (teks dot peen FDFDH)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        processed_img = cv2.addWeighted(enhanced, 2.0, blur, -1.0, 0)

    return img, processed_img

def read_numbers_with_ocr(image_path, mode="BEARING"):
    """
    Membaca angka/teks menggunakan EasyOCR berdasarkan mode pengujian.
    """
    print(f"Memproses gambar: {image_path} (Mode: {mode})...")
    
    reader = easyocr.Reader(['en'], gpu=True) 
    original_img, processed_img = preprocess_image(image_path, mode)
    h, w = processed_img.shape[:2]

    if mode == "BEARING":
        allowlist = '12345' # Hanya boleh menebak angka 1 sampai 5
        text_threshold = 0.15
        low_text = 0.2
        mag_ratio = 1.5
        min_prob = 0.2
    else:
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        text_threshold = 0.1
        low_text = 0.3
        mag_ratio = 1.0
        min_prob = 0.3

    # --- 1. Deteksi teks Horizontal (Orientasi Normal) ---
    results_normal = reader.readtext(processed_img, detail=1, allowlist=allowlist,
                                     text_threshold=text_threshold, low_text=low_text, 
                                     mag_ratio=mag_ratio, link_threshold=0.8, width_ths=0.2)

    # --- 2. Deteksi teks Vertikal (Putar 90 Derajat CCW) ---
    processed_img_ccw = cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    results_ccw = reader.readtext(processed_img_ccw, detail=1, allowlist=allowlist,
                                  text_threshold=text_threshold, low_text=low_text, 
                                  mag_ratio=mag_ratio, link_threshold=0.8, width_ths=0.2)

    all_results = []
    
    # Kumpulkan hasil horizontal
    for (bbox, text, prob) in results_normal:
        if prob > min_prob:
            all_results.append((bbox, text, prob))

    # Kumpulkan dan petakan hasil vertikal
    for (bbox, text, prob) in results_ccw:
        if prob > min_prob:
            original_bbox = []
            for point in bbox:
                xr, yr = point
                x = w - yr
                y = xr
                original_bbox.append([x, y])
            all_results.append((original_bbox, text, prob))

    print(f"Ditemukan {len(all_results)} teks gabungan.")

    # Gambar kotak (bounding box) dan teks hasil deteksi
    for (bbox, text, prob) in all_results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        tl = (int(min(x_coords)), int(min(y_coords)))
        br = (int(max(x_coords)), int(max(y_coords)))

        cv2.rectangle(original_img, tl, br, (0, 255, 0), 2)
        text_to_display = f"{text} ({prob:.2f})"
        cv2.putText(original_img, text_to_display, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        print(f"Terbaca: {text} | Akurasi: {prob:.2f}")

    height, width = original_img.shape[:2]
    max_height = 800
    if height > max_height:
        scaling_factor = max_height / float(height)
        original_img = cv2.resize(original_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    cv2.imshow("Hasil Deteksi OCR", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # PILIH MODE PENGUJIAN: "BEARING" atau "ENGINE"
    MODE_PENGUJIAN = "ENGINE"
    
    # Path gambar (silakan disesuaikan)
    TEST_IMAGE_PATH = r"A:\ASTRAtech\Lomba\T-MIND\SOPGuardAI\20Juta\Trial OCR\ocr3.jpg" 
    
    try:
        read_numbers_with_ocr(TEST_IMAGE_PATH, mode=MODE_PENGUJIAN)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
