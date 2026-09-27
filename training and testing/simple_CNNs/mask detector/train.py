import numpy as np 
import keras 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory as loader 
train_ds =  loader(
    directory = '',
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
                loss='categorical_crossentropy',
                metrics=[keras.metrics.Recall(name='recall')])

print(model.summary())

history = model.fit (train_ds,epochs=2,vaidation_data=test_ds)

from keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, 
           show_layer_names=True)

from emotions_detector.train import ploting
ploting(history)