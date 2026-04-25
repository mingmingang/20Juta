import cv2
from ultralytics import YOLO
import mysql.connector
from datetime import datetime
import paho.mqtt.client as mqtt
import json
import time
from flask import Flask, Response
import threading

# Load YOLO model
model = YOLO("c:\\KULIAH\\SEMESTER 4\\IOT\\PROJEK FIX\\PROGRAM PYTHON\\last (8).pt")
print("Class Names:", model.names)

# Koneksi ke MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smart_ac_db"
)
cursor = db.cursor()

# Konfigurasi MQTT
mqtt_broker = "astratech.id"
mqtt_port = 1883
mqtt_topic = "smartac/detection"
mqtt_buzzer_topic = "smartac/buzzer"

mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)

# Flask untuk live stream
app = Flask(__name__)
cap = cv2.VideoCapture(1)

# Variabel global untuk anotasi frame
annotated_frame = None
last_db_save_time = time.time()

def yolo_loop():
    global annotated_frame, last_db_save_time
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        boxes = results[0].boxes.data

        person_count = 0
        helmet_count = 0
        no_helmet_count = 0
        glasses_count = 0
        no_glasses_count = 0

        for box in boxes:
            cls_id = int(box[-1])
            class_name = model.names[cls_id].lower().replace(" ", "")
            if class_name == "people":
                person_count += 1
            elif class_name == "helmet":
                helmet_count += 1
            elif class_name == "nohelmet":
                no_helmet_count += 1
            elif class_name == "glasses":
                glasses_count += 1
            elif class_name == "noglasses":
                no_glasses_count += 1

        annotated_frame = results[0].plot()

        # Tampilkan info deteksi di frame
        cv2.putText(annotated_frame, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Helmets: {helmet_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"No Helmets: {no_helmet_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(annotated_frame, f"Glasses: {glasses_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"No Glasses: {no_glasses_count}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = time.time()

        # menyimpan ke database setiap 7 detik
        if current_time - last_db_save_time >= 7:
            sql = """
                INSERT INTO person_count 
                (count, helmets, no_helmet, glasses, no_glasses, timestamp) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            val = (person_count, helmet_count, no_helmet_count, glasses_count, no_glasses_count, now)
            cursor.execute(sql, val)
            db.commit()
            last_db_save_time = current_time
            print("Saved to MySQL:", val)

        # kirim data ke MQTT
        payload = {
            "people": person_count,
            "helmets": helmet_count,
            "no_helmet": no_helmet_count,
            "glasses": glasses_count,
            "no_glasses": no_glasses_count,
            "timestamp": now
        }
        mqtt_client.publish(mqtt_topic, json.dumps(payload))
        print(f"Published to MQTT ({mqtt_topic}):", payload)

        # buzzer
        buzzer_on = no_helmet_count > 0 or no_glasses_count > 0
        if buzzer_on:
            mqtt_client.publish(mqtt_buzzer_topic, "BUZZER_ON")
        else:
            mqtt_client.publish(mqtt_buzzer_topic, "BUZZER_OFF")

# fungsi untuk MJPEG streaming
def generate_stream():
    global annotated_frame
    while True:
        if annotated_frame is not None:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# menjalankan YOLO di thread terpisah
threading.Thread(target=yolo_loop, daemon=True).start()

# flask untuk video stream
app.run(host='0.0.0.0', port=5000)

# bersihkan semua koneksi saat program berhenti
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    db.close()
    mqtt_client.disconnect()

import atexit
atexit.register(cleanup)
