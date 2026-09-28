# note :
# model is trained on kaggle because of the limited GPU resources 
# on my local machine.
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import pandas as pd
from keras.models import Model
from keras.layers import (Input, Dense, Conv2D, MaxPooling2D, Flatten,
                          RandomFlip, RandomRotation, RandomZoom, Rescaling)

folder_path = '/kaggle/input/utkface-new/UTKFace'

age = []
gender = []
img_path = []

for file in os.listdir(folder_path):
    parts = file.split('_')
    if len(parts) >= 3:
        age.append(int(parts[0]))      
        gender.append(int(parts[1]))   
        img_path.append(os.path.join(folder_path, file)) 

train_df = pd.DataFrame({'age': age, 'gender': gender, 'img_path': img_path})
print("Dataset Shape:", train_df.shape)
print(train_df.head(3))

train_df_split = train_df.sample(frac=0.8, random_state=42)
test_df_split = train_df.drop(train_df_split.index)

def load_and_preprocess_image(path, age_label, gender_label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image, {'age_output': tf.cast(age_label, tf.float32), 
                   'gender_output': tf.cast(gender_label, tf.float32)}


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df_split['img_path'].values, train_df_split['age'].values, train_df_split['gender'].values)
)
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df_split['img_path'].values, test_df_split['age'].values, test_df_split['gender'].values)
)
test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# --- MODEL ARCHITECTURe ---
input_layer = Input(shape=(224, 224, 3))

# Data Augmentation & Normalization Pipeline
x = RandomFlip("horizontal")(input_layer)
x = RandomRotation(0.1)(x)
x = RandomZoom(0.1)(x)
x = Rescaling(1./255)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

flatten_layer = Flatten(name='flatten')(x)

age_dense = Dense(128, activation='relu', name='age_dense')(flatten_layer)
age_output = Dense(1, activation='linear', name='age_output')(age_dense)

gender_dense = Dense(128, activation='relu', name='gender_dense')(flatten_layer)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_dense)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
print(model.summary())

model.compile(optimizer='adam',
              loss={'age_output': 'mae', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': ['mae'], 'gender_output': ['accuracy']})

# Train using tf.data objects instead of DataFrames
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)


from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Total Loss")
plt.plot(history.history["val_loss"], label="Val Total Loss")
plt.title("Model Total Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history["gender_output_accuracy"], label="Train Gender Accuracy")
plt.plot(history.history["val_gender_output_accuracy"], label="Val Gender Accuracy")
plt.title("Gender Branch Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()
