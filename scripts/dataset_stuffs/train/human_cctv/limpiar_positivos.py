import os
import cv2

# --- Configuración ---
IMAGES_DIR = "images"          # Imágenes originales completas
LABELS_DIR = "labelTxt"        # Anotaciones originales .txt
POSITIVOS_DIR = "positivos"    # Carpeta de recortes positivos
UMBRAL_AREA = 0.20             # 20% del área mínima para aceptar como humano

# --- Eliminar recortes irrelevantes ---
eliminados = 0
total = 0

for nombre_img in os.listdir(POSITIVOS_DIR):
    if not nombre_img.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    ruta_recorte = os.path.join(POSITIVOS_DIR, nombre_img)
    base = "_".join(nombre_img.split("_")[:-2])  # ej: imagen_0_2.jpg → imagen
    fila = int(nombre_img.split("_")[-2])
    col = int(nombre_img.split("_")[-1].split(".")[0])

    ruta_img_original = os.path.join(IMAGES_DIR, base + ".jpg")
    ruta_txt = os.path.join(LABELS_DIR, base + ".txt")

    if not os.path.exists(ruta_img_original) or not os.path.exists(ruta_txt):
        continue

    # Leer imagen original
    img = cv2.imread(ruta_img_original)
    if img is None:
        continue

    # Redimensionar como se hizo antes
    h, w = img.shape[:2]
    size = min(h, w)
    img = cv2.resize(img, (size, size))
    h = w = size
    gw, gh = w // 4, h // 4
    gx, gy = col * gw, fila * gh
    area_recorte = gw * gh

    # Leer anotaciones
    personas = []
    with open(ruta_txt, 'r') as f:
        for line in f:
            datos = line.strip().split()
            if len(datos) < 9:
                continue
            pts = list(map(float, datos[:8]))
            x_coords = pts[::2]
            y_coords = pts[1::2]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            personas.append([x1, y1, x2, y2])

    # Revisar cuánto ocupa cada persona
    mantener = False
    for x1, y1, x2, y2 in personas:
        # Intersección con recorte
        ix1 = max(x1, gx)
        iy1 = max(y1, gy)
        ix2 = min(x2, gx + gw)
        iy2 = min(y2, gy + gh)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        if inter_area / area_recorte >= UMBRAL_AREA:
            mantener = True
            break

    # Si ninguna persona ocupa suficiente área, eliminar imagen
    total += 1
    if not mantener:
        os.remove(ruta_recorte)
        eliminados += 1

print(f"🧹 Total positivos revisados: {total}")
print(f"🗑️ Eliminados por área insuficiente: {eliminados}")
print(f"✅ Restantes válidos: {total - eliminados}")