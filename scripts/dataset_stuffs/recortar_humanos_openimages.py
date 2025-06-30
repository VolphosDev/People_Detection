import os
import pandas as pd
import cv2

# 📂 Rutas
carpeta_imagenes = 'train/human'
csv_anotaciones = 'train-annotations-bbox.csv'
output_dir = 'train/human'
os.makedirs(output_dir, exist_ok=True)

# 🧠 Clase "persona" en Open Images
CLASE_PERSONA = '/m/01g317'

# 📄 Leer CSV
df = pd.read_csv(csv_anotaciones)
df_personas = df[df['LabelName'] == CLASE_PERSONA]

# 👥 Agrupar por imagen
agrupado = df_personas.groupby('ImageID')

total_recortes = 0
imagenes_procesadas = set()

for image_id, group in agrupado:
    image_path = os.path.join(carpeta_imagenes, image_id + '.jpg')
    if not os.path.exists(image_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    recortes_hechos = 0

    for _, row in group.iterrows():
        x_min = int(row['XMin'] * w)
        x_max = int(row['XMax'] * w)
        y_min = int(row['YMin'] * h)
        y_max = int(row['YMax'] * h)

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            continue

        if width < 128 or height < 128:
            continue  # ❌ Demasiado pequeño, lo descartamos

        recorte = img[y_min:y_max, x_min:x_max]
        filename = f"persona_open_{total_recortes}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), recorte)
        recortes_hechos += 1
        total_recortes += 1

    # ✅ Si hubo al menos un recorte válido, eliminamos la imagen original
    # ❌ Si no hubo ningún recorte, también la eliminamos
    os.remove(image_path)

print(f"✅ Se guardaron {total_recortes} recortes mayores a 128x128 en: {output_dir}")