# V2.2 ArUco + YOLO + MediaPipe Pose Integration

## Problem

Saat ada **2+ orang** di depan kamera, MediaPipe Pose tidak bisa membedakan **siapa** yang skeleton-nya di-track. Sering terjadi skeleton "lompat" ke orang lain. Kita butuh cara memastikan skeleton hanya di-draw untuk **operator tertentu** (yang punya ArUco marker).

## Solution: YOLO Person → ArUco ID → Cropped Pose

```
┌──────────────────────────────┐
│  Frame dari kamera           │
│                              │
│  ┌─────┐      ┌─────┐       │
│  │ P1  │      │ P2  │       │  ← YOLO deteksi semua "person"
│  │ArUco│      │     │       │
│  │ID=0 │      │     │       │  ← ArUco match ke YOLO bbox P1
│  └─────┘      └─────┘       │
│     ↓                        │
│  Crop P1 → MediaPipe Pose   │  ← Pose HANYA dari crop P1
│  → Skeleton di-remap ke     │
│    koordinat frame asli      │
└──────────────────────────────┘
```

**Pipeline per frame:**
1. **YOLO** → deteksi semua orang → bounding box per person
2. **ArUco** → deteksi marker → dapatkan posisi center marker
3. **Match** → ArUco center jatuh di dalam YOLO bbox mana? → itu operator-nya
4. **Crop** → potong frame sesuai bbox operator
5. **MediaPipe Pose** → jalankan pose estimation HANYA di crop
6. **Remap** → konversi koordinat skeleton dari crop → frame asli
7. **Draw** → gambar skeleton + bbox + nama operator

## Proposed Changes

### [NEW] [V2.2 Coba Person Tracking Use ArUco.py](file:///D:/Documents%20HDD/Kuli-ah/comp/T-MIND/03%20Prototype/First%20Trial%20(Snitching%203%20CV)/V2.2%20Coba%20Person%20Tracking%20Use%20ArUco.py)

Satu file script, config di atas, berisi:
- **Config section**: kamera, YOLO model, ArUco dict, ID_MAP, pose settings
- **ArUcoDetector class**: dari V2.1 (tidak diubah)
- **YOLO person detection**: `ultralytics` YOLOv8n, filter class `person` saja
- **MediaPipe Holistic**: dari referensi [Mediapipe_fromFile_Video.py](file:///D:/Documents%20HDD/Kuli-ah/comp/T-MIND/03%20Prototype/First%20Trial%20%28Snitching%203%20CV%29/Mediapipe_fromFile_Video.py), dijalankan pada **cropped region** saja
- **Matching logic**: cek apakah ArUco center berada di dalam YOLO bbox
- **Skeleton remapping**: koordinat pose landmark di-remap dari crop ke frame asli
- **HUD**: FPS, operator info, tracking status

**Dependencies**: `ultralytics`, `mediapipe`, `opencv-python`, `numpy`

## Verification Plan

### Manual Verification
1. Jalankan script: `.venv\Scripts\python.exe "V2.2 Coba Person Tracking Use ArUco.py"`
2. Arahkan ArUco marker (ID 0) ke kamera
3. Verifikasi:
   - YOLO bbox muncul di semua orang (kotak biru)
   - ArUco marker terdeteksi (kotak hijau)
   - Skeleton HANYA muncul di orang yang pegang ArUco
   - Nama operator tampil di HUD
4. Tes dengan 2 orang: skeleton harus tetap di operator, bukan lompat ke orang lain
