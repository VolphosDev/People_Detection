from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pickle

# Configuración
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# Generador de datos con validación (20%)
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Generadores para entrenamiento y validación
train_gen = datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

# Imprimir clases
print("Clases:", train_gen.class_indices)
print(f"Human: {np.sum(train_gen.classes == 0)}, No Human: {np.sum(train_gen.classes == 1)}")

# Modelo CNN simple
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilación
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks útiles
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Entrenamiento con validación
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Guardar modelo y entrenamiento
model.save("modelo_cnn_final.h5")
with open("historial_simple.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Entrenamiento completo con validación.")