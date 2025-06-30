import os
import json
import cv2

# Rutas
carpeta_imagenes = 'train_uttx/train'
archivo_anotaciones = os.path.join(carpeta_imagenes, '_annotations.coco.json')
output_dir = 'train/human'
os.makedirs(output_dir, exist_ok=True)

# Leer anotaciones COCO
with open(archivo_anotaciones, 'r') as f:
    coco = json.load(f)

# Crear diccionarios útiles
id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
id_to_category = {cat['id']: cat['name'] for cat in coco['categories']}

# Iterar por las anotaciones
for i, annot in enumerate(coco['annotations']):
    category_id = annot['category_id']
    category_name = id_to_category[category_id]
    
    # Solo personas
    if category_name.lower() != 'person':
        continue

    image_id = annot['image_id']
    bbox = annot['bbox']  # formato COCO: [x, y, width, height]
    x, y, w, h = map(int, bbox)

    image_path = os.path.join(carpeta_imagenes, id_to_filename[image_id])
    if not os.path.exists(image_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    # Recortar y guardar
    recorte = img[y:y+h, x:x+w]
    filename = f"persona_{i}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), recorte)

print(f"✅ Recortes de personas guardados en: {output_dir}")