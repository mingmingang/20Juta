"""
Generate Checkerboard Pattern untuk Kalibrasi Kamera
====================================================
Menghasilkan file PNG checkerboard 10x7 kotak (= 9x6 inner corners)
yang bisa dicetak di kertas A4.

Cara pakai:
  1. Jalankan script ini → akan generate 'checkerboard_9x6.png'
  2. Print file PNG tersebut di kertas A4
  3. Tempelkan di permukaan datar (kardus/papan)
  4. Gunakan untuk kalibrasi kamera di aplikasi V2.9
"""

import numpy as np
import cv2
import os

# Konfigurasi — sesuaikan dengan CALIB_BOARD_SIZE di V2.9
INNER_CORNERS_X = 9   # jumlah inner corner horizontal
INNER_CORNERS_Y = 6   # jumlah inner corner vertikal

# Ukuran kotak dalam piksel (untuk print A4 landscape, 60px ≈ 2cm)
SQUARE_SIZE_PX = 80

# Hitung ukuran total
COLS = INNER_CORNERS_X + 1   # 10 kolom kotak
ROWS = INNER_CORNERS_Y + 1   # 7 baris kotak

# Margin (border putih di sekeliling)
MARGIN = 60

# Ukuran image total
img_w = COLS * SQUARE_SIZE_PX + 2 * MARGIN
img_h = ROWS * SQUARE_SIZE_PX + 2 * MARGIN

# Buat image putih
img = np.ones((img_h, img_w), dtype=np.uint8) * 255

# Gambar kotak-kotak hitam
for row in range(ROWS):
    for col in range(COLS):
        if (row + col) % 2 == 0:
            x1 = MARGIN + col * SQUARE_SIZE_PX
            y1 = MARGIN + row * SQUARE_SIZE_PX
            x2 = x1 + SQUARE_SIZE_PX
            y2 = y1 + SQUARE_SIZE_PX
            img[y1:y2, x1:x2] = 0  # hitam

# Simpan
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "checkerboard_9x6.png")
cv2.imwrite(output_path, img)

print(f"Checkerboard berhasil dibuat!")
print(f"  File  : {output_path}")
print(f"  Ukuran: {img_w} x {img_h} px")
print(f"  Grid  : {COLS} x {ROWS} kotak ({INNER_CORNERS_X}x{INNER_CORNERS_Y} inner corners)")
print(f"\nCara pakai:")
print(f"  1. Print file ini di kertas A4 (landscape)")
print(f"  2. Tempel di permukaan DATAR (kardus/papan)")
print(f"  3. Jalankan V2.9, tekan 'k' untuk mode kalibrasi")
print(f"  4. Arahkan checkerboard ke kamera dari berbagai sudut")
print(f"  5. Tekan SPACE untuk capture (min 12x)")
print(f"  6. Tekan 'k' untuk selesai")
