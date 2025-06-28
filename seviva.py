import os
import pytz
import threading
import cv2
import numpy as np
import torch
from datetime import datetime
import warnings
from flask import Flask, request, Response, render_template

app = Flask(__name__)

# Variables globales
latest_detected_frame = None
latest_raw_frame = None
frame_lock = threading.Lock()
prev_img = None
detection_happened = False  # Bandera para saber si hubo detección

# Cargar el modelo YOLOv5 solo una vez
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # Solo personas

# Suprimir advertencias
warnings.simplefilter("ignore", category=FutureWarning)

# Crear carpeta para capturas
os.makedirs('capturas', exist_ok=True)

# Variables para manejar las carpetas de tiempo
last_capture_time = None

def detect_and_draw(img):
    global last_capture_time, detection_happened
    print("Analizando imagen...")

    results = model(img)
    save_capture = False
    person_count = 0
    high_confidence_count = 0

    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf >= 0.25:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            person_count += 1
        if conf >= 0.75:
            save_capture = True
            high_confidence_count += 1
            detection_happened = True

    cv2.putText(img, f'Posibles personas en pantalla: {person_count}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if high_confidence_count > 0:
        cv2.putText(img, f'Personas detectadas en pantalla: {high_confidence_count}',
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    now = datetime.now(pytz.timezone('America/Lima'))
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")
    minute_str = str((now.minute // 10) * 10).zfill(2)

    day_folder = os.path.join('capturas', date_str)
    os.makedirs(day_folder, exist_ok=True)

    time_folder = os.path.join(day_folder, f'{hour_str}-{minute_str}')
    os.makedirs(time_folder, exist_ok=True)

    if save_capture:
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(time_folder, f'persona_{timestamp}.jpg')
        cv2.imwrite(filename, img)

    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

@app.route('/upload', methods=['POST'])
def upload():
    global latest_detected_frame, latest_raw_frame, prev_img
    frame_bytes = request.data

    nparr = np.frombuffer(frame_bytes, np.uint8)
    curr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if curr_img is None:
        return 'Invalid image data', 400

    _, raw_jpeg = cv2.imencode('.jpg', curr_img)
    with frame_lock:
        latest_raw_frame = raw_jpeg.tobytes()

    if prev_img is not None:
        small_prev_img = cv2.resize(prev_img, (200, 150))
        small_curr_img = cv2.resize(curr_img, (200, 150))

        diff = cv2.absdiff(small_prev_img, small_curr_img)
        print("Diff mean:", np.mean(diff))
        if np.mean(diff) < 2.2:
            print('Skipped - Frame too similar')
        return 'Skipped - Frame too similar', 200

    prev_img = curr_img.copy()

    with frame_lock:
        latest_detected_frame = detect_and_draw(curr_img)

    return 'OK', 200

@app.route('/check_detection')
def check_detection():
    global detection_happened
    if detection_happened:
        detection_happened = False  # Resetear la bandera después de avisar
        return {'detected': True}
    else:
        return {'detected': False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        while True:
            with frame_lock:
                frame = latest_detected_frame or latest_raw_frame
                if not frame:
                    continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 
