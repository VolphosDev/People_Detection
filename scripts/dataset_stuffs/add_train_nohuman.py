import os
import shutil
import random
from tqdm import tqdm

# 📁 Carpetas
origen_base = 'train_nohuman_stuffs/PASS_dataset'
destino = 'train/no_human'
os.makedirs(destino, exist_ok=True)

# 🔢 Parámetros
max_imagenes = 25000
extensiones_validas = ('.jpg', '.jpeg', '.png')

# 🔍 Recolectar rutas de imágenes
todas_las_imagenes = []

for carpeta_id in range(20):
    carpeta_actual = os.path.join(origen_base, str(carpeta_id))
    if not os.path.isdir(carpeta_actual):
        continue
    for archivo in os.listdir(carpeta_actual):
        if archivo.lower().endswith(extensiones_validas):
            todas_las_imagenes.append(os.path.join(carpeta_actual, archivo))

# 🎲 Mezclar y seleccionar
random.shuffle(todas_las_imagenes)
seleccionadas = todas_las_imagenes[:max_imagenes]

# 📤 Copiar imágenes con barra de progreso
print("📦 Copiando imágenes...")
for i, ruta in enumerate(tqdm(seleccionadas, desc="Copiando")):
    nombre = f"nohuman_{i}.jpg"
    shutil.copy(ruta, os.path.join(destino, nombre))

print(f"\n✅ Se copiaron {len(seleccionadas)} imágenes a {destino}")

# 📊 Conteo final
human_dir = 'train/human'
nohuman_dir = 'train/no_human'
print(f"\n📊 Conteo final:")
print("Human:", len(os.listdir(human_dir)))
print("No Human:", len(os.listdir(nohuman_dir)))