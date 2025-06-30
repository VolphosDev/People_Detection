import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hiperparámetros
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
AUG_DIR = "train_augmented"

def augment_dataset(original_dir="train", output_dir="train_augmented"):
    if os.path.exists(output_dir):
        print(f"{output_dir} ya existe. Se usará tal cual.")
        return

    print("Creando dataset aumentado (solo flip a positivos)...")
    os.makedirs(output_dir, exist_ok=True)
    classes = os.listdir(original_dir)

    for class_name in classes:
        input_class_dir = os.path.join(original_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for fname in os.listdir(input_class_dir):
            img_path = os.path.join(input_class_dir, fname)
            img = Image.open(img_path).convert("RGB")

            img.save(os.path.join(output_class_dir, fname))

            if class_name.lower() == "human":
                base_name = os.path.splitext(fname)[0]
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped.save(os.path.join(output_class_dir, f"{base_name}_flip.jpg"))

augment_dataset()

# 2️⃣ --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# 3️⃣ --- Datasets ---
train_data = datasets.ImageFolder("train_augmented", transform=transform)
val_data = datasets.ImageFolder("../val_human_cctv", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# 4️⃣ --- Modelo CNN ---
class DetectorCNN(nn.Module):
    def __init__(self):
        super(DetectorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48x48
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 24x24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 12x12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 5️⃣ --- Entrenamiento ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct_train, total_train = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"🧪 Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).squeeze().long()
        correct_train += (preds == labels.squeeze().long()).sum().item()
        total_train += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train * 100

    # Evaluación
    model.eval()
    val_loss = 0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).squeeze().long()
            correct_val += (preds == labels.squeeze().long()).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val * 100

    print(f"📊 Epoch {epoch+1}/{EPOCHS}")
    print(f"  📉 Train Loss: {avg_train_loss:.4f} | 🧠 Train Acc: {train_acc:.2f}%")
    print(f"  ✅ Val   Loss: {avg_val_loss:.4f} | 🎯 Val Acc : {val_acc:.2f}%\n")

torch.save(model.state_dict(), "human_cnn_final.pth")
print("💾 Modelo guardado como human_cnn_final.pth")