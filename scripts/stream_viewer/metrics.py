import os
import time
import warnings
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
MODEL_PATH = "../dataset_stuffs/train/human_cctv/modelo_detector.pth"
IMAGE_FOLDER = "../dataset_stuffs/test/images"
LABEL_FOLDER = "../dataset_stuffs/test/labelTxt"

# Modelo
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

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Modelo cargado")

# Obtener lista de imágenes una sola vez
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))])

# Repeticiones
NUM_RUNS = 10
precision_total = 0
recall_total = 0
fps_total = 0

for run in range(NUM_RUNS):
    y_true = []
    y_pred = []
    start_time = time.time()

    for filename in image_files:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(filename)[0] + ".txt")

        img = cv2.imread(image_path)
        if img is None or not os.path.exists(label_path):
            continue

        has_human = False
        patch_predictions = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_name = parts[8]
                if cls_name.lower() != "person":
                    continue
                has_human = True
                coords = list(map(float, parts[:8]))
                pts = np.array(coords, dtype=np.int32).reshape((4, 2))
                x, y, w, h = cv2.boundingRect(pts)
                recorte = img[y:y+h, x:x+w]
                if recorte.shape[0] < 10 or recorte.shape[1] < 10:
                    continue
                tensor_img = transform(recorte).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(tensor_img).item()
                patch_predictions.append(pred)

        final_pred = 1 if any(p >= 0.90 for p in patch_predictions) else 0
        y_true.append(1 if has_human else 0)
        y_pred.append(final_pred)

    elapsed = time.time() - start_time
    fps = len(y_true) / elapsed if elapsed > 0 else 0

    precision_total += precision_score(y_true, y_pred)
    recall_total += recall_score(y_true, y_pred)
    fps_total += fps

# Promedios finales
precision_avg = (precision_total / NUM_RUNS) * 100
recall_avg = (recall_total / NUM_RUNS) * 100
fps_avg = fps_total / NUM_RUNS

print(f"\n📊 Promedios sobre {NUM_RUNS} ejecuciones y {len(image_files)} imágenes:")
print(f"✅ Precisión promedio: {precision_avg:.2f}%")
print(f"✅ Recall promedio:    {recall_avg:.2f}%")
print(f"⚡ FPS promedio:       {fps_avg:.2f}")