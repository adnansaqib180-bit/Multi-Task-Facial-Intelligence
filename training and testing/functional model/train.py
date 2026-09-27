from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os 
import pandas as pd

folder_path = 'training and testing/functional model/'

age=[]
gender=[]
img_path=[]
for file in os.listdir(folder_path):
    age.append(file.split('_')[0])
    gender.append(file.split('_')[1])
    img_path.append(file)

train_df = pd.DataFrame({'age':age,'gender':gender,'img_path':img_path})
print(train_df.shape)
print(train_df.head(3))
train = train_df.sample(frac=0.8,random_state=42)
test = train_df.drop(train.index)
generated_train_df = ImageDataGenerator(rescale=1./255,
                                           rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
generated_test_df = ImageDataGenerator(rescale=1./255)

train_generator = generated_train_df.flow_from_dataframe(dataframe=train,
                                                        directory=folder_path,
                                                        x_col='img_path',
                                                        y_col=['age', 'gender'],
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='multi_output')

test_generator = generated_test_df.flow_from_dataframe(dataframe=test,
                                                      directory=folder_path,
                                                      x_col='img_path',
                                                      y_col=['age', 'gender'],
                                                      target_size=(224, 224),
                                                      batch_size=32,
                                                      class_mode='multi_output')

input_layer = Input(shape=(224, 224, 3))

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
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

# --- THE SPLIT POINT ---
flatten_layer = Flatten(name='flatten')(x)


# --- AGE PREDICTION TRACK (Regression) ---
age_dense = Dense(128, activation='relu', name='age_dense')(flatten_layer)
age_output = Dense(1, activation='linear', name='age_output')(age_dense)


# --- GENDER PREDICTION TRACK (Binary Classification) ---
gender_dense = Dense(128, activation='relu', name='gender_dense')(flatten_layer)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_dense)



model = Model(inputs=input_layer, outputs=[age_output, gender_output], name='vgg16_split_heads')

print(model.summary())

model.compile(optimizer='adam',
              loss={'age_output': 'mae', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

history = model.fit(train_generator, epochs=10, validation_data=test_generator)






