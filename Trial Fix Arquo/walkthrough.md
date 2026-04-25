# Walkthrough: Sequence Tracker untuk Perakitan Bearing

Saya telah berhasil memperbarui file `V2.4 SOP - Milk Box Bearing Sequence.py` dengan logika *sequence* (urutan) yang sudah Anda jelaskan.

## Perubahan yang Dilakukan

1. **Memastikan Urutan Sesuai Arahan**: 
   Pola yang diminta dimodelkan dalam sistem `SEQUENCE_STEPS`:
   - Ambil Storage #1, #3, #5
   - Taruh Jig #5, #3, #1
   - Ambil Storage #2, #4
   - Taruh Jig #4, #2
2. **Debouncing Detections (Anti-flicker)**:
   Proses deteksi YOLO tidak akan "melompat" walau kamera sempat blur sebentar. Program menunggu sekitar 8 *frame* untuk mengonfirmasi bahwa Storage/Jig memang "Terisi" atau "Kosong", sehingga sistem jauh lebih stabil.
3. **Peringatan Salah Urutan (Robust Warning)**:
   Setiap saat kita sedang diminta *action* tertentu (misal "`Ambil Storage #3`"), sistem sekaligus mengecek apakah ada Storage atau Jig yang berubah tidak sesuai perintah (misalnya tidak sengaja mengambil Storage #5 duluan). Jika hal ini terjadi, HUD di layar akan memunculkan tulisan `SALAH URUTAN!` berwarna merah lengkap dengan instruksi kotak mana yang harus dikembalikan.
4. **Sequence HUD interaktif**:
   Sekarang terdapat layar instruksi (kotak panel di kanan atas kamera) yang membimbing langkah demi langkah, dan juga otomatis muncul ketika tepat ada 10 Bounding Box (Storage & Jig) yang terdeteksi dari konfigurasi `V2.3` (`roi_config.json`).
5. **Tombol "R" untuk Mereset**:
   Apabila ada kegagalan selama proses atau ingin diulangi dari awal, Anda cukup menekan tombol `R` pada *keyboard*, dan urutan akan dimulai ulang dari pemeriksaan "Storage Penuh, Jig Kosong".

> [!TIP]
> **Cara Menjalankan**: 
> 1. Seperti biasa, pastikan 10 ROI sudah dibuat/diload (Tekan `L` untuk me-load *bounding box* jika belum otomatis meload). 
> 2. Tekan `ENTER` untuk memulai deteksi. 
> 3. Baca instruksi dari box **SEQUENCE STATUS** di layar kanan atas Anda.
> 4. Anda juga bisa menekan tombol `R` kapan saja untuk mengulang proses instruksinya dari nol (kembali ke tahap pengecekan Setup).
