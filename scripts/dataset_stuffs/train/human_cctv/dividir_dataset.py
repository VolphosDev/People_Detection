import os
import shutil
import random

# --- Configuración ---
ORIGEN_POS = "positivos"
ORIGEN_NEG = "negativos"
DEST_TRAIN = "train"
DEST_VAL = "../val_human_cctv"  # Se crea al mismo nivel que human_cctv
RATIO_VAL = 0.2  # 20% validación

def preparar_carpetas():
    for base in [DEST_TRAIN, DEST_VAL]:
        for clase in ["positivos", "negativos"]:
            os.makedirs(os.path.join(base, clase), exist_ok=True)

def dividir_y_mover(origen, destino_train, destino_val, ratio_val=0.2):
    archivos = [f for f in os.listdir(origen) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(archivos)

    n_val = int(len(archivos) * ratio_val)
    val_set = set(archivos[:n_val])

    for archivo in archivos:
        src = os.path.join(origen, archivo)
        if archivo in val_set:
            dst = os.path.join(destino_val, archivo)
        else:
            dst = os.path.join(destino_train, archivo)
        shutil.move(src, dst)  # ← ahora se MUEVE, no copia

    return len(archivos) - n_val, n_val

# --- Ejecutar ---
preparar_carpetas()

train_pos, val_pos = dividir_y_mover(
    ORIGEN_POS,
    os.path.join(DEST_TRAIN, "positivos"),
    os.path.join(DEST_VAL, "positivos"),
    RATIO_VAL
)

train_neg, val_neg = dividir_y_mover(
    ORIGEN_NEG,
    os.path.join(DEST_TRAIN, "negativos"),
    os.path.join(DEST_VAL, "negativos"),
    RATIO_VAL
)

# --- Resumen ---
print("✅ División completada con éxito.")
print(f"🟢 Positivos - Train: {train_pos}, Val: {val_pos}")
print(f"⚫ Negativos - Train: {train_neg}, Val: {val_neg}")