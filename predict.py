import joblib as jb 
import numpy as np
cnn=jb.load('brain_tumor_detection_cnn_model.h5')

import tensorflow as tf
from keras.preprocessing import image
test_image = tf.keras.utils.load_img('prediction/yes1.jpg',target_size=(224,224))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)

print(result)

if result[0][0]==1:
    print("yes")
else:
    print("no")