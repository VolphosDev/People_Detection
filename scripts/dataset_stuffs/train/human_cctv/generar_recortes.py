import os
import cv2

IMAGES_DIR = "images"
LABELS_DIR = "labelTxt"
SALIDA_POSITIVOS = "positivos"
SALIDA_NEGATIVOS = "negativos"
REJILLAS = 4

os.makedirs(SALIDA_POSITIVOS, exist_ok=True)
os.makedirs(SALIDA_NEGATIVOS, exist_ok=True)

total_positivos = 0
total_negativos = 0

def box_in_cell(box, gx, gy, gw, gh):
    x1, y1, x2, y2 = box
    return not (x2 < gx or x1 > gx + gw or y2 < gy or y1 > gy + gh)

def generar_recortes(im_path, txt_path, salida_pos, salida_neg, rejillas=4):
    global total_positivos, total_negativos

    img = cv2.imread(im_path)
    if img is None:
        print(f"No se pudo leer la imagen {im_path}")
        return

    h, w = img.shape[:2]
    size = min(h, w)
    img = cv2.resize(img, (size, size))
    h = w = size
    gw, gh = w // rejillas, h // rejillas

    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                datos = line.strip().split()
                if len(datos) < 9:
                    continue
                pts = list(map(float, datos[:8]))
                x_coords = pts[::2]
                y_coords = pts[1::2]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                boxes.append([x1, y1, x2, y2])

    base = os.path.splitext(os.path.basename(im_path))[0]

    for row in range(rejillas):
        for col in range(rejillas):
            gx, gy = col * gw, row * gh
            recorte = img[gy:gy+gh, gx:gx+gw]
            tiene_persona = any(box_in_cell(b, gx, gy, gw, gh) for b in boxes)

            carpeta = salida_pos if tiene_persona else salida_neg
            nombre = f"{base}_{row}_{col}.jpg"
            cv2.imwrite(os.path.join(carpeta, nombre), recorte)

            # Contar
            if tiene_persona:
                total_positivos += 1
            else:
                total_negativos += 1

# --- Procesar todo el dataset ---
for nombre_img in os.listdir(IMAGES_DIR):
    if not nombre_img.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    ruta_img = os.path.join(IMAGES_DIR, nombre_img)
    ruta_txt = os.path.join(LABELS_DIR, os.path.splitext(nombre_img)[0] + ".txt")

    generar_recortes(ruta_img, ruta_txt, SALIDA_POSITIVOS, SALIDA_NEGATIVOS, REJILLAS)

# --- Mostrar resultados ---
print("✅ ¡Recortes generados y clasificados!")
print(f"📦 Total positivos (con humanos): {total_positivos}")
print(f"📦 Total negativos (sin humanos): {total_negativos}")