import pandas as pd

# Cargar anotaciones completas
df = pd.read_csv('train-annotations-bbox.csv')

# IDs con personas
person_ids = set(df[df['LabelName'] == '/m/01g317']['ImageID'].unique())

# Todos los IDs (en el set de entrenamiento)
all_ids = df['ImageID'].unique()

# Filtrar los que NO contienen personas
no_person_ids = [img_id for img_id in all_ids if img_id not in person_ids]

# Tomar solo los primeros 10000
no_person_ids = no_person_ids[:10000]

# Guardar en archivo para descarga
with open('no_person_10000.txt', 'w') as f:
    for img_id in no_person_ids:
        f.write(f'train/{img_id}\n')

print(f"[✅] Lista generada con {len(no_person_ids)} imágenes sin personas.")
