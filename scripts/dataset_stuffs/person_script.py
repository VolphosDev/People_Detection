import pandas as pd

# Cargar CSV de bounding boxes
df = pd.read_csv('train-annotations-bbox.csv')

# Filtrar clase 'person'
df_person = df[df['LabelName'] == '/m/01g317']

# Tomar solo los primeros 5,000 IDs únicos
image_ids = df_person['ImageID'].unique()[:20000]

# Guardar en archivo de lista para descarga
with open('person_20000.txt', 'w') as f:
    for img_id in image_ids:
        f.write(f'train/{img_id}\n')

print(f"[✅] Lista generada con {len(image_ids)} imágenes con personas.")