from flask import json
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
from flask import Flask, Response, request, jsonify 
from flask_cors import CORS
import pyodbc
import qrcode
import base64
from io import BytesIO
from ultralytics import YOLO

# ==================== SQL Server Setup ====================
DB_CONFIG = (
    "Driver={SQL Server};"
    "Server=AANG;" # Contoh: localhost atau .\SQLEXPRESS
    "Database=TMIND_DB;"
    "Trusted_Connection=yes;"
)

# ============================================================
# KONFIGURASI ASLI KAMU (100% Sesuai Request)
# ============================================================
MODEL_PATH = "efficientdet_lite0.tflite"
MAX_RESULTS = 10           
SCORE_THRESHOLD = 0.35     
CAMERA_INDEX = 0           
FRAME_WIDTH = 1280         
FRAME_HEIGHT = 720         

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]
TARGET_OBJECTS = []  

# BEARING YOLO
BEARING_MODEL_PATH = "A:/ASTRAtech/Lomba/T-MIND/SOPGuardAI/20Juta/Trial OCR/best (22).pt"
URL_IP_KAMERA = "http://admin:admin@192.168.100.100:8081/video"
model_bearing = YOLO(BEARING_MODEL_PATH)

HAND_MODEL_PATH = "A:/ASTRAtech/Lomba/T-MIND/SOPGuardAI/20Juta/Trial Fix Arquo/hand_landmarker.task"
ZONES_FILE = "tray_config.json"
TOUCH_HOLD_FRAMES = 8

app = Flask(__name__)
CORS(app) 

latest_detection_result = None

# Tambahkan folder export di atas
EXPORT_DIR = r"A:\ASTRAtech\Lomba\T-MIND\SOPGuardAI\Export_Generator"
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# FUNGSI UNTUK GENERATE FISIK PNG ARUCO
def generate_and_save_aruco(marker_id, full_name):
    try:
        # Menggunakan DICT_4X4_50 sesuai kriteria lomba kamu
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500) # Ukuran 500px
        
        # Buat nama file: ArUco_ID0_SAHAR_ROMANSA.png
        clean_name = full_name.replace(" ", "_").upper()
        filename = f"ArUco_ID{marker_id}_{clean_name}.png"
        filepath = os.path.join(EXPORT_DIR, filename)
        
        cv2.imwrite(filepath, marker_img)
        print(f"[INFO] ArUco Ter-generate: {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal generate ArUco: {e}")
        return False

def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def save_aruco_marker(marker_id, name):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500)
        filename = f"ArUco_ID{marker_id}_{name.replace(' ', '_')}.png"
        filepath = os.path.join(EXPORT_DIR, filename)
        cv2.imwrite(filepath, marker_img)
        return filename
    except Exception as e:
        print(f"Error Gen ArUco: {e}")
        return None

def download_model():
    if os.path.exists(MODEL_PATH): return True
    print("[INFO] Mengunduh model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        return True
    except Exception as e:
        print(f"[ERROR] Gagal unduh model: {e}")
        return False

def get_color(index):
    return COLORS[index % len(COLORS)]

def draw_detection(frame, detection, color):
    bbox = detection.bounding_box
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    corner_len = min(30, w // 4, h // 4)
    corner_thick = 4
    cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thick)
    cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thick)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thick)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thick)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thick)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thick)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick)

    for i, category in enumerate(detection.categories):
        label = f"{category.category_name}: {category.score:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
        label_y = max(y - 10 - (i * (text_h + 10)), text_h + 5)
        cv2.rectangle(frame, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 5), color, -1)
        cv2.putText(frame, label, (x + 5, label_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def draw_hud(frame, fps, detection_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objek Terdeteksi: {detection_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    filter_text = f"Filter: {', '.join(TARGET_OBJECTS[:3])}" if TARGET_OBJECTS else "Filter: Semua Objek"
    cv2.putText(frame, filter_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def detection_callback(result, output_image, timestamp_ms):
    global latest_detection_result
    latest_detection_result = result

# ============================================================
# LOGIKA GENERATOR STREAMING
# ============================================================

def generate_frames():
    global latest_detection_result
    download_model()
    
    # Inisialisasi MediaPipe
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        max_results=MAX_RESULTS,
        score_threshold=SCORE_THRESHOLD,
        result_callback=detection_callback,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    category_color_map = {}
    color_index = 0

    with mp.tasks.vision.ObjectDetector.create_from_options(options) as detector:
        timestamp = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp += 1
            detector.detect_async(mp_image, timestamp)

            detection_count = 0
            if latest_detection_result:
                for detection in latest_detection_result.detections:
                    category_name = detection.categories[0].category_name
                    if TARGET_OBJECTS and category_name not in TARGET_OBJECTS: continue
                    
                    if category_name not in category_color_map:
                        category_color_map[category_name] = get_color(color_index)
                        color_index += 1
                    
                    draw_detection(frame, detection, category_color_map[category_name])
                    detection_count += 1

            frame_count += 1
            if (time.time() - start_time) >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            draw_hud(frame, fps, detection_count)

            # Convert frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield dalam format MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        # Query mengambil data dari tabel sesuai kolom di SSMS kamu
        query = """
            SELECT username, fullName, division, employeeId, monitoringHours, securityStatus, accessLevel 
            FROM Users 
            WHERE username=? AND password=?
        """
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Map hasil database (tuple) ke JSON Object
            return jsonify({
                "success": True, 
                "userData": {
                    "username": str(user[0]),
                    "fullName": str(user[1]),
                    "division": str(user[2]),
                    "employeeId": str(user[3]),
                    "monitoringHours": str(user[4]),
                    "securityStatus": str(user[5]),
                    "accessLevel": str(user[6])
                }
            })
        else:
            return jsonify({"success": False, "message": "User tidak ditemukan!"}), 401

    except Exception as e:
        print(f"Error Database: {e}")
        return jsonify({"success": False, "message": "Terjadi kesalahan server"}), 500
    
@app.route('/api/update-profile', methods=['POST'])
def update_profile():
    data = request.json
    username = data.get('username') # Kunci utama
    full_name = data.get('fullName')
    division = data.get('division')
    employee_id = data.get('employeeId')

    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            UPDATE Users 
            SET fullName = ?, division = ?, employeeId = ?
            WHERE username = ?
        """
        cursor.execute(query, (full_name, division, employee_id, username))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Profil berhasil diperbarui"})
    except Exception as e:
        print(f"Error Update DB: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Route untuk mengambil semua user
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id, fullName, employeeId, division, accessLevel, securityStatus, username, arucoId FROM Users")
        rows = cursor.fetchall()
        
        users = []
        for r in rows:
            users.append({
                "id": r[0], "fullName": r[1], "employeeId": r[2],
                "division": r[3], "accessLevel": r[4], "status": r[5], "username": r[6], "arucoId": r[7]
            })
        conn.close()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.json
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(arucoId) FROM Users")
        max_id_row = cursor.fetchone()
        new_aruco_id = 0 if max_id_row[0] is None else max_id_row[0] + 1

        query = """
            INSERT INTO Users (username, password, fullName, employeeId, division, accessLevel, securityStatus, arucoId)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            data['username'], data['password'], data['fullName'], 
            data['employeeId'], data['division'], data['accessLevel'], 'OFFLINE', new_aruco_id
        ))
        
        generate_and_save_aruco(new_aruco_id, data['fullName'])

        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True, 
            "message": "User & ArUco Marker berhasil dibuat!",
            "arucoId": new_aruco_id
        })
    except Exception as e:
        print(f"Error add user: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        

@app.route('/api/users/update', methods=['POST'])
def update_user_management():
    data = request.json
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            UPDATE Users 
            SET fullName = ?, employeeId = ?, division = ?, accessLevel = ?, securityStatus = ?
            WHERE id = ?
        """
        cursor.execute(query, (
            data['fullName'], data['employeeId'], data['division'], 
            data['accessLevel'], data['status'], data['id']
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate-qr', methods=['POST'])
def api_generate_qr():
    data = request.json
    qr_data = data.get('data', 'T-MIND Default')
    filename = data.get('filename', f"QR_{int(time.time())}")

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    # Simpan ke folder
    filepath = os.path.join(EXPORT_DIR, f"{filename}.png")
    img.save(filepath)

    return jsonify({
        "success": True,
        "image": image_to_base64(img), # Kirim datanya supaya tampil di React
        "path": filepath
    })

# ==================== API GENERATE ARUCO ====================
@app.route('/api/generate-aruco', methods=['POST'])
def api_generate_aruco():
    data = request.json
    marker_id = int(data.get('id', 0))
    dict_type = data.get('dict', "4x4_50")
    size = int(data.get('size', 500))
    filename = data.get('filename', f"ArUco_ID{marker_id}")

    # Logic OpenCV ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Simpan ke folder
    filepath = os.path.join(EXPORT_DIR, f"{filename}.png")
    cv2.imwrite(filepath, marker_img)

    # Convert ke PIL untuk di-encode ke Base64
    from PIL import Image
    img_pil = Image.fromarray(marker_img).convert("RGB")

    return jsonify({
        "success": True,
        "image": image_to_base64(img_pil),
        "path": filepath
    })

# ==================== API GENERATOR KHUSUS BEARING ====================
def generate_checkbearing_frames():
    # Menggunakan alamat IP Camera
    cap_bearing = cv2.VideoCapture(URL_IP_KAMERA)
    
    # Tambahkan timeout atau retry jika kamera IP butuh waktu untuk tersambung
    if not cap_bearing.isOpened():
        print("[ERROR] Gagal tersambung ke IP Kamera Sahar. Pastikan HP di-connect wifi yang sama.")
        return

    while True:
        success, frame = cap_bearing.read()
        if not success:
            # Jika stream terputus sebentar, coba sambung lagi (khusus IP Camera)
            cap_bearing.release()
            cap_bearing = cv2.VideoCapture(URL_IP_KAMERA)
            continue

        # Inference menggunakan YOLOv11
        # Kita set verbose=False agar terminal Python tidak berisik
        results = model_bearing(frame, verbose=False)
        
        # Gambarkan box & label secara otomatis (pake .plot() sesuai kodemu)
        annotated_frame = results[0].plot()

        # Convert ke JPEG untuk di-stream ke browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==================== ENDPOINT API BARU ====================
@app.route('/checkbearing_feed')
def checkbearing_feed():
    """Endpoint untuk stream live hasil deteksi Bearing di dashboard"""
    return Response(generate_checkbearing_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


picking_events_log = []

class HandPickingDetector:
    def __init__(self):
        # Load Config Zona dari JSON
        self.zones = []
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE, 'r') as f:
                self.zones = json.load(f)
        
        # Init MediaPipe
        base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.timestamp = 0
        self.touch_counters = {}
        self.touch_triggered = {}

    def process(self, frame):
        h, w = frame.shape[:2]
        self.timestamp += 33 # Simulasi 30 FPS
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(mp_image, self.timestamp)
        
        active_touches = []
        if res.hand_landmarks:
            for i, hand_lm in enumerate(res.hand_landmarks):
                # Ambil koordinat ujung jari (ID 8 = Jari Telunjuk)
                fingertips = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                index_finger = fingertips[8]
                
                # Cek terhadap zona tray
                for idx, zone in enumerate(self.zones):
                    pts = np.array(zone['points'], dtype=np.int32)
                    if pts.size > 0:
                        dist = cv2.pointPolygonTest(pts, index_finger, False)
                        if dist >= 0: # Di dalam zona
                            active_touches.append(idx)
                            self.touch_counters[idx] = self.touch_counters.get(idx, 0) + 1
                            
                            # Jika disentuh lebih dari n frame, masukkan ke Log
                            if self.touch_counters[idx] >= TOUCH_HOLD_FRAMES and not self.touch_triggered.get(idx, False):
                                msg = f"Mengambil di {zone['name']}"
                                picking_events_log.insert(0, {"time": time.strftime("%H:%M:%S"), "msg": msg})
                                if len(picking_events_log) > 10: picking_events_log.pop()
                                self.touch_triggered[idx] = True
                            
                            # Gambar UI Touch
                            cv2.fillPoly(frame, [pts], (0, 255, 100)) # Hijau

        # Reset trigger jika tangan sudah dilepas
        for k in list(self.touch_counters.keys()):
            if k not in active_touches:
                self.touch_counters[k] = 0
                self.touch_triggered[k] = False

        # Gambar zona poligon (Idle)
        for zone in self.zones:
            pts = np.array(zone['points'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (200, 200, 200), 2)
            if pts.size > 0:
                cv2.putText(frame, zone['name'], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Inisialisasi
picking_sys = HandPickingDetector()

# ==================== WORK SEQUENCE & INCIDENT DATA ====================
# Simulasi database / state monitoring
work_steps = [
    {"id": 1, "name": "DETEKSI ENGINE BLOCK ID", "status": "done"},
    {"id": 2, "name": "IDENTIFIKASI BEARING CODE", "status": "active", "desc": "Mendapatkan kombinasi huruf A, B, C."},
    {"id": 3, "name": "PENEMPATAN BEARING #1", "status": "wait"},
]

incident_logs = [
    {"type": "CRITICAL", "time": "14:22", "title": "MISMATCH BEARING #1", "msg": "Operator memasang bearing nomor 3 ke slot nomor 2.", "color": "#f43f5e"},
    {"type": "WARNING", "time": "14:15", "title": "BLOK MESIN BURAM", "msg": "Kamera kesulitan membaca kode huruf karena oli.", "color": "#f59e0b"}
]

@app.route('/api/work-sequence')
def get_work_sequence():
    return jsonify(work_steps)

@app.route('/api/incidents')
def get_incidents():
    return jsonify(incident_logs)

# ==================== ENDPOINT VIDEO STREAM ====================
def generate_picking_frames():
    cap_hand = cv2.VideoCapture(0) # Sesuaikan index kamera
    while True:
        success, frame = cap_hand.read()
        if not success: break
        
        # Proses deteksi picking
        picking_sys.process(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/picking_feed')
def picking_feed():
    return Response(generate_picking_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== ENDPOINT JSON DATA (Log Pesan) ====================
@app.route('/api/picking_logs')
def get_picking_logs():
    return jsonify(picking_events_log)
    
if __name__ == '__main__':
    # Ganti port dari 5000 ke 5001
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)