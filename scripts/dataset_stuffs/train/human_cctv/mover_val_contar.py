import os
import shutil
import random

# --- Configuración ---
ORIGEN = os.path.join("train", "positivos")
DESTINO = os.path.join("..", "val_human_cctv", "positivos")
RATIO = 0.2  # 20%

# --- Asegurar que la carpeta de destino exista ---
os.makedirs(DESTINO, exist_ok=True)

# --- Listar y seleccionar imágenes ---
imagenes = [f for f in os.listdir(ORIGEN) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(imagenes)
cantidad_val = int(len(imagenes) * RATIO)
seleccionadas = imagenes[:cantidad_val]

# --- Mover imágenes seleccionadas ---
for nombre in seleccionadas:
    ruta_origen = os.path.join(ORIGEN, nombre)
    ruta_destino = os.path.join(DESTINO, nombre)
    shutil.move(ruta_origen, ruta_destino)

# --- Función para contar imágenes ---
def contar_imagenes(ruta):
    if not os.path.exists(ruta):
        return 0
    return len([f for f in os.listdir(ruta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# --- Rutas a contar ---
train_pos = os.path.join("train", "positivos")
train_neg = os.path.join("train", "negativos")
val_pos = os.path.join("..", "val_human_cctv", "positivos")
val_neg = os.path.join("..", "val_human_cctv", "negativos")

# --- Mostrar resumen ---
print("✅ Movimiento completado.")
print(f"🟢 Total imágenes movidas: {cantidad_val}")
print(f"📁 Desde: {ORIGEN}")
print(f"📁 Hacia: {DESTINO}")
print("\n📊 Conteo actual de imágenes:")
print(f"   🟩 train/positivos       : {contar_imagenes(train_pos)}")
print(f"   ⬛ train/negativos       : {contar_imagenes(train_neg)}")
print(f"   🟩 val_human_cctv/positivos : {contar_imagenes(val_pos)}")
print(f"   ⬛ val_human_cctv/negativos : {contar_imagenes(val_neg)}")