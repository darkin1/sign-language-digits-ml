import tensorflow as tf
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('./models/sign-10-40-epochs-1550868468')


IMG_SIZE = 64
DATADIR = "./dataset_raw/2"
# DATADIR = "./predict_images/prawa_reka"
# DATADIR = "./predict_images/lewa_reka"
# DATADIR = "./predict_images/bledne"
# DATADIR = "./predict_images/prawa_reka_jasne"
# DATADIR = "./predict_images/lewa_reka_jasne"
# DATADIR = "./predict_images/lewa_reka_jasne_mirror"
FILENAME = "IMG_4225.JPG"

def load_image():
    X = []
    img_array = cv2.imread(os.path.join(DATADIR, FILENAME), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    X.append(new_array)
    X = 1-np.array(X).astype('float32')/255.
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
    
    return X

def show_image(X):
    plt.imshow(np.squeeze(X), cmap="gray")
    plt.show()


def predict(model, X):
    predicted = model.predict(X)
    # print(predicted)

    predicted_class_number = model.predict_classes(X)
    # print(predicted_class_number)

    print("Classes probability=%s" % (predicted))
    print(" === THE BEST PREDICTED NUMBER: %s === " % (predicted_class_number))

# TODO: 
# wybrać 3 najlepsze klasy i przeliczać na procenty
# wyuczyć model na większej ilości epok



### RUN 

X = load_image()
predict(model, X)
show_image(X)