import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory as loader 
train_ds =  loader(
    directory = 'Multi-Task-Facial-Intelligence/data/train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(48, 48)
)
test_ds =  loader(
    directory = 'Multi-Task-Facial-Intelligence/data/test',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(48,48)
)

model =  Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
def build_model(hp):
    model = Sequential()
    nodes = hp.Int('nodes',min_value= 20,max_value=80,step= 8)
    activation = hp.Choice('activation',values=['tanh','sigmoid','relu','elu'])
    num_layers = hp.Int('layers',min_value=1,max_value=3)
    for i in range (num_layers):
        model.add(Dense(units=nodes,activation=activation))
    model.add(Dense(7,activation='softmax'))          

    return model
model.compile(optimizer='adam',
                loss='categorical_phrase_entropy',
                metrics=[keras.metrics.Recall(name='recall')])

print(model.summary())
history = model.fit (train_ds,epochs=2,vaidation_data=test_ds)

def ploting(data):
    history = data
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history["recall"], label="Train Recall")
    plt.plot(history.history["val_recall"], label="Validation Recall")
    plt.title("Model Recall")
    plt.ylabel("Recall")
    plt.xlabel("Epoch")
    plt.legend()


    plt.tight_layout()
    plt.show()

    #saving the model

# model.save('final_ann.keras')
# print('model saved')