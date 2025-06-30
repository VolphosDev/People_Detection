import os
import threading
import cv2
import numpy as np
import warnings
from datetime import datetime
from flask import Flask, Response, render_template
import torch
import torch.nn as nn
from torchvision import transforms
import time

app = Flask(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

frame_lock = threading.Lock()
latest_frame = None
output_folder = "figuras_detectadas"
os.makedirs(output_folder, exist_ok=True)

# Modelo PyTorch
class DetectorCNN(nn.Module):
    def __init__(self):
        super(DetectorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)
model.load_state_dict(torch.load("../dataset_stuffs/train/human_cctv/human_cnn_final.pth", map_location=device))
model.eval()
print("✅ Modelo CNN PyTorch cargado.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def crear_sustractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

bg_subtractor = crear_sustractor()
ultimo_reinicio = time.time()
ultimo_guardado = 0
frames_guardados_en_segundo = 0
ultimo_segundo = int(time.time())

VIDEO_SOURCE = 0
cap = cv2.VideoCapture(VIDEO_SOURCE)

def procesar_video():
    global latest_frame, bg_subtractor, ultimo_reinicio
    global ultimo_guardado, frames_guardados_en_segundo, ultimo_segundo

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Imprimir resolución del frame
        print(f"[FRAME] Resolución recibida: {frame.shape[1]}x{frame.shape[0]}")

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        movimiento_total = np.sum(fg_mask == 255)
        umbral_cambio_global = frame.shape[0] * frame.shape[1] * 0.6

        if movimiento_total > umbral_cambio_global and (time.time() - ultimo_reinicio > 2):
            print("⚠️ Cambio global detectado. Reiniciando fondo.")
            bg_subtractor = crear_sustractor()
            ultimo_reinicio = time.time()
            continue

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x_min, y_min, x_max, y_max = frame.shape[1], frame.shape[0], 0, 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            ancho_total = x_max - x_min
            alto_total = y_max - y_min
            if ancho_total > frame.shape[1] * 0.9 and alto_total > frame.shape[0] * 0.9:
                print("🚫 Movimiento en todo el frame: posible movimiento de cámara. Ignorando frame.")
                continue

        result = frame.copy()
        humanos_detectados = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            recorte = frame[y:y+h, x:x+w]
            tensor_img = transform(recorte).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor_img).item()

            if pred >= 0.5:
                label = "Humano"
                color = (0, 255, 0)
                humanos_detectados += 1
            else:
                label = "No humano"
                color = (0, 0, 255)

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, f"{label} ({pred:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ahora = datetime.now()
        texto_fecha = ahora.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, texto_fecha, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(result, f"Personas: {humanos_detectados}", (result.shape[1] - 160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        segundo_actual = int(time.time())
        if segundo_actual != ultimo_segundo:
            frames_guardados_en_segundo = 0
            ultimo_segundo = segundo_actual

        if humanos_detectados > 0 and (time.time() - ultimo_guardado >= 0.5) and frames_guardados_en_segundo < 2:
            bloque_tiempo = ahora.strftime("%Y%m%d-%H%M") + f"{(ahora.second // 15) * 15:02d}"
            carpeta_actual = os.path.join(output_folder, bloque_tiempo)
            os.makedirs(carpeta_actual, exist_ok=True)

            timestamp = ahora.strftime("%Y%m%d-%H%M%S-%f")
            filename = os.path.join(carpeta_actual, f"frame_{timestamp}.jpg")
            cv2.imwrite(filename, result)

            ultimo_guardado = time.time()
            frames_guardados_en_segundo += 1
            print(f"Guardado: {filename}")

        _, jpeg = cv2.imencode('.jpg', result)
        with frame_lock:
            latest_frame = jpeg.tobytes()

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

if __name__ == '__main__':
    threading.Thread(target=procesar_video, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
