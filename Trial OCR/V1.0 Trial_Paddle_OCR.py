import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

# ============================================================
#   OCR ENGINE BLOCK — PaddleOCR v3
#   Khusus membaca kode dot peen (FDFDH) pada engine block
# ============================================================

def preprocess(image_path):
    """
    Preprocessing ringan tanpa crop untuk dot peen pada logam.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")

    h, w = img.shape[:2]
    print(f"[INFO] Ukuran gambar: {w}x{h}")

    # ── Kembalikan Crop Area (Pojok Kanan Bawah) ──
    # Teks dot peen ukurannya sangat kecil, jadi harus dicrop
    Y_START = 0.80
    Y_END   = 1.00
    X_START = 0.60
    X_END   = 1.00

    crop = img[int(h*Y_START):int(h*Y_END),
               int(w*X_START):int(w*X_END)]

    # Ubah ke Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Perbesar (Upscale 4x). Cukup untuk OCR tapi tidak merusak seperti 8x.
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # CLAHE ringan untuk meratakan pencahayaan logam tanpa membuatnya terlalu putih
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    sharp_gray = clahe.apply(big)

    # Kembalikan ke format 3 Channel (BGR) untuk PaddleOCR
    sharp_bgr = cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)
    
    # Buat juga versi inverted (warna dibalik)
    inverted_bgr = cv2.cvtColor(cv2.bitwise_not(sharp_gray), cv2.COLOR_GRAY2BGR)

    return img, crop, sharp_gray, sharp_bgr, inverted_bgr


def correct_dotpeen(text):
    """
    Koreksi karakter yang sering salah pada dot peen logam.
    Format kode: F-D-F-D-H (posisi 0-4)
      posisi 1,3 → harusnya D, sering terbaca O atau 0
      posisi 4   → harusnya H, sering terbaca K
    """
    res = list(text.upper())
    for i, ch in enumerate(res):
        if i in (1, 3) and ch in ('O', '0'):
            res[i] = 'D'
        if i == 4 and ch == 'K':
            res[i] = 'H'
    return ''.join(res)


def read_engine_code(image_path, save_output=True):
    """
    Pipeline lengkap: crop → preprocess → PaddleOCR → hasil.
    """
    print("=" * 55)
    print("  PaddleOCR v3 — Engine Block Code Reader")
    print("=" * 55)

    original, crop, sharp_gray, sharp_bgr, inverted_bgr = preprocess(image_path)

    # Inisialisasi PaddleOCR v3
    print("[INFO] Memuat PaddleOCR...")
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        device='cpu'
    )

    all_results = []

    # ── OCR versi normal ──
    print("[INFO] OCR versi normal...")
    try:
        for res in ocr.predict(sharp_bgr):
            if res is None:
                continue
            for item in res:
                text = item.get('rec_text', '')
                prob = item.get('rec_score', 0)
                bbox = item.get('dt_polys', [])
                # Filter dihapus sebagian agar teks apa saja yang terdeteksi bisa tampil di layar
                if prob >= 0.1:
                    corrected = correct_dotpeen(text)
                    all_results.append({
                        'source':     'normal',
                        'raw':        text,
                        'text':       corrected,
                        'confidence': prob,
                        'bbox':       bbox
                    })
    except Exception as e:
        print(f"[WARN] OCR normal error: {e}")

    # ── OCR versi inverted ──
    print("[INFO] OCR versi inverted...")
    try:
        for res in ocr.predict(inverted_bgr):
            if res is None:
                continue
            for item in res:
                text = item.get('rec_text', '')
                prob = item.get('rec_score', 0)
                bbox = item.get('dt_polys', [])
                if prob >= 0.1:
                    corrected = correct_dotpeen(text)
                    all_results.append({
                        'source':     'inverted',
                        'raw':        text,
                        'text':       corrected,
                        'confidence': prob,
                        'bbox':       bbox
                    })
    except Exception as e:
        print(f"[WARN] OCR inverted error: {e}")

    # ── Tampilkan hasil ──
    all_results.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\n{'='*55}")
    print(f"  HASIL OCR")
    print(f"{'='*55}")
    print(f"{'No':<4}{'Source':<12}{'Raw':<15}{'Corrected':<15}{'Conf':>6}")
    print("-" * 52)

    if all_results:
        for i, r in enumerate(all_results, 1):
            print(f"  {i:<3}{r['source']:<12}{r['raw']:<15}"
                  f"{r['text']:<15}{r['confidence']:.2f}")
        print(f"\n>>> Kode terbaca: {all_results[0]['text']}")
    else:
        print("  Tidak ada teks terdeteksi.")
        print("\n  Saran:")
        print("  - Foto lebih dekat ke area kode (jarak 15-20cm)")
        print("  - Pencahayaan dari samping (raking light)")
        print("  - Kamera tegak lurus ke permukaan")
        print("  - Sesuaikan Y_START, X_START di fungsi preprocess()")

    # ── Visualisasi ──
    vis = cv2.resize(crop, None, fx=1, fy=1, interpolation=cv2.INTER_LANCZOS4)

    # Gambar bbox pada visualisasi
    if all_results:
        crop_h, crop_w = crop.shape[:2]
        proc_h, proc_w = sharp_gray.shape[:2]
        # Hitung skala
        sx = (crop_w * 1) / proc_w
        sy = (crop_h * 1) / proc_h

        for r in all_results[:5]:
            if len(r['bbox']) == 0:
                continue
            pts = np.array(r['bbox'], dtype=np.float32)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts = pts.astype(np.int32)

            color = (0, 255, 0)   if r['confidence'] >= 0.6 else \
                    (0, 165, 255) if r['confidence'] >= 0.35 else \
                    (0, 0, 255)

            cv2.polylines(vis, [pts], True, color, 2)
            label = f"{r['text']} ({r['confidence']:.2f})"
            cv2.putText(vis, label,
                        (pts[0][0], max(pts[0][1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    # Simpan hasil
    if save_output:
        base     = os.path.splitext(image_path)[0]
        out_path = f"{base}_hasil_paddle.jpg"
        cv2.imwrite(out_path, vis)
        print(f"\n[INFO] Hasil disimpan: {out_path}")

    # Tampilkan jendela
    cv2.imshow("Crop Area (4x)", vis)
    cv2.imshow("Processed (grayscale)", cv2.resize(sharp_gray, None, fx=0.5, fy=0.5))
    
    print("[INFO] Tekan sembarang tombol untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_results


# ============================================================
#   MAIN
# ============================================================
if __name__ == "__main__":

    # ── Ganti path ini sesuai lokasi foto kamu ──
    IMAGE_PATH = r"C:\KULIAH\SEMESTER 6\T-MIND\05 Project TMMIN\Trial OCR\OCR4.jpeg"

    try:
        results = read_engine_code(IMAGE_PATH, save_output=True)
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {IMAGE_PATH}")
    except Exception as e:
        print(f"[ERROR] {e}")