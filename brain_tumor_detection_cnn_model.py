import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import joblib as jb
import keras 
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory('train',target_size=(224,224),batch_size=32,shuffle=True,class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('test',target_size=(224,224),batch_size=16,shuffle=False,class_mode='binary')

cnn_model = Sequential()

cnn_model.add(Conv2D(filters=224 , kernel_size=3 , activation='relu' , input_shape=[224,224,3]))
cnn_model.add(MaxPool2D(pool_size=2,strides=2))
cnn_model.add(Conv2D(filters=224 , kernel_size=3 , activation='relu' ))
cnn_model.add(MaxPool2D(pool_size=2 , strides=2))
cnn_model.add(Dropout(0.5))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=128, activation='relu'))
cnn_model.add(Dense(units=1 , activation='sigmoid'))
cnn_model.compile(optimizer = 'Adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
cnn_model.fit(x = training_set , validation_data = test_set , epochs = 10)

jb.dump(cnn_model,'brain_tumor_detection_model.h5')

test_image = tf.keras.utils.load_img('prediction/yes3.jpg',target_size=(224,224))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn_model.predict(test_image)
training_set.class_indices
print(result)

if result[0][0] == 1:
    print('yes')
else:
    print('no')





