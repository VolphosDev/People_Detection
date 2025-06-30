import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from datetime import datetime

# Rutas de los modelos
MODEL_PATHS = {
    "model_human_cnn_final": "../dataset_stuffs/train/human_cctv/human_cnn_final.pth",
    "model_modelo_detector": "../dataset_stuffs/train/human_cctv/modelo_detector.pth"
}

# Red CNN
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

# Transformación
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def crear_sustractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

def procesar_con_modelo(nombre_modelo, modelo_path, video_path):
    print(f"🔍 Procesando con modelo: {nombre_modelo}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectorCNN().to(device)
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_name = f"{nombre_modelo}_output.mp4"
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    bg_subtractor = crear_sustractor()
    frame_count = 0
    detecciones_altas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy()
        humanos_detectados = 0

        for cnt in contours:
            if cv2.contourArea(cnt) < 10000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            recorte = frame[y:y+h, x:x+w]
            try:
                tensor_img = transform(recorte).unsqueeze(0).to(device)
            except Exception:
                continue

            with torch.no_grad():
                pred = model(tensor_img).item()

            if pred >= 0.9:
                humanos_detectados += 1

                if pred >= 0.98:
                    detecciones_altas += 1

                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, f"Humano ({pred:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(result, f"No humano ({pred:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(result, f"Frame: {frame_count}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result, f"Detectados: {humanos_detectados}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(result)

    cap.release()
    out.release()

    print(f"📈 FPS total: {fps}")
    print(f"✅ Total frames: {frame_count}")
    print(f"🟢 Detecciones con confianza >= 0.98: {detecciones_altas}")
    print(f"📁 Video guardado: {output_name}\n")

# MAIN
if __name__ == "__main__":
    video_path = "video_prueba.mp4"
    for nombre, path in MODEL_PATHS.items():
        procesar_con_modelo(nombre, path, video_path)