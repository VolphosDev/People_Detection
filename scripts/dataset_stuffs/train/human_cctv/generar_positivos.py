import os
import cv2

IMAGES_DIR = "images"
LABELS_DIR = "labelTxt"
SALIDA_POSITIVOS = os.path.join("train", "positivos")  # Carpeta dentro de 'train'
TAMANO = 96

os.makedirs(SALIDA_POSITIVOS, exist_ok=True)

total_positivos = 0

def generar_recortes_positivos(im_path, txt_path, salida_pos):
    global total_positivos

    img = cv2.imread(im_path)
    if img is None:
        print(f"No se pudo leer la imagen {im_path}")
        return

    if not os.path.exists(txt_path):
        print(f"No se encontró anotación para {im_path}")
        return

    base = os.path.splitext(os.path.basename(im_path))[0]

    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            datos = line.strip().split()
            if len(datos) < 9:
                continue

            pts = list(map(float, datos[:8]))
            x_coords = pts[::2]
            y_coords = pts[1::2]
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))

            # Validar límites
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # Recorte y redimensión
            recorte = img[y1:y2, x1:x2]
            if recorte.size == 0:
                continue
            recorte_resized = cv2.resize(recorte, (TAMANO, TAMANO))

            nombre = f"{base}_person_{i}.jpg"
            cv2.imwrite(os.path.join(salida_pos, nombre), recorte_resized)
            total_positivos += 1

for nombre_img in os.listdir(IMAGES_DIR):
    if not nombre_img.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    ruta_img = os.path.join(IMAGES_DIR, nombre_img)
    ruta_txt = os.path.join(LABELS_DIR, os.path.splitext(nombre_img)[0] + ".txt")

    generar_recortes_positivos(ruta_img, ruta_txt, SALIDA_POSITIVOS)

# --- Resultados ---
print("✅ ¡Recortes positivos generados desde las cajas anotadas!")
print(f"📦 Total positivos: {total_positivos}")