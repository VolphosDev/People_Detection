import os
import random

carpeta_human = 'train/human'
imagenes = os.listdir(carpeta_human)

limite = 30000 

if len(imagenes) > limite:
    excedente = len(imagenes) - limite
    print(f"Eliminando {excedente} imágenes para balancear...")

    random.shuffle(imagenes)
    a_eliminar = imagenes[:excedente]

    for img in a_eliminar:
        os.remove(os.path.join(carpeta_human, img))

    print(f"Ahora se tiene {limite} imágenes en train/human")
else:
    print("Ya estás balanceado o no hay exceso.")
