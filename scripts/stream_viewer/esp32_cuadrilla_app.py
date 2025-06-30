import os
import threading
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
import time

app = Flask(__name__)

frame_lock = threading.Lock()
latest_frame = None
output_folder = "figuras_detectadas"
os.makedirs(output_folder, exist_ok=True)

# Modelo PyTorch
class DetectorCNN(nn.Module):
    def __init__(self):
        super(DetectorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 96x96 -> 48x48
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48x48 -> 24x24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24x24 -> 12x12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),                      # ← Tamaño correcto
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)
model.load_state_dict(torch.load('../dataset_stuffs/train/human_cctv/modelo_detector.pth', map_location=device))
model.eval()
print("✅ Modelo CNN PyTorch cargado.")

# Preprocesamiento PyTorch
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Inicializar sustractor de fondo
def crear_sustractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

bg_subtractor = crear_sustractor()
ultimo_reinicio = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame, bg_subtractor, ultimo_reinicio

    file_bytes = request.get_data()
    if not file_bytes:
        return "❌ No se recibió imagen", 400

    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "❌ Imagen no válida", 400

    # --- Sustracción de fondo ---
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # 🔁 Detección de cambio global
    movimiento_total = np.sum(fg_mask == 255)
    umbral_cambio_global = frame.shape[0] * frame.shape[1] * 0.6  # 60% del frame

    if movimiento_total > umbral_cambio_global and (time.time() - ultimo_reinicio > 2):
        print("⚠️ Cambio global detectado. Reiniciando fondo.")
        bg_subtractor = crear_sustractor()
        ultimo_reinicio = time.time()
        return "⚠️ Fondo reiniciado", 200

    # --- Análisis de contornos ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        recorte = frame[y:y+h, x:x+w]
        tensor_img = transform(recorte).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor_img).item()

        label = "Humano" if pred >= 0.5 else "No humano"
        color = (0, 255, 0) if label == "Humano" else (0, 0, 255)

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, f"{label} ({pred:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label == "Humano":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            filename = os.path.join(output_folder, f"humano_{timestamp}_{count}.jpg")
            cv2.imwrite(filename, recorte)

        count += 1

    _, jpeg = cv2.imencode('.jpg', result)
    with frame_lock:
        latest_frame = jpeg.tobytes()

    return "✅ Imagen procesada", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)