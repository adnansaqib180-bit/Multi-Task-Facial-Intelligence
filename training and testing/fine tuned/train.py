import matplotlib.pyplot as plt
import keras 
from keras import layers, models
from keras.utils import image_dataset_from_directory as loader 
from keras.applications import EfficientNetB0

train_ds = loader(
    directory=r'c:\Users\USER\OneDrive\Desktop\train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",         
    batch_size=32,
    image_size=(224, 224)     
)

test_ds = loader(
    directory=r'c:\Users\USER\OneDrive\Desktop\test',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",         
    batch_size=32,
    image_size=(224, 224)     
)

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3), 
    pooling=None               
)
base_model.trainable = False   

inputs = layers.Input(shape=(224, 224, 3)) 
x = base_model(inputs, training=False)      
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print('--- Phase 1: Training Top Layers oonly ---')
print(model.summary())
model.fit(train_ds, epochs=7, validation_data=test_ds)

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n-- Phase 2 : Fine-Tuning Last 20 Layers ---")
print(model.summary())  

model.fit(train_ds, epochs=50, validation_data=test_ds)
