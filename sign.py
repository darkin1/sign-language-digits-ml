import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#https://www.kaggle.com/ardamavi/sign-language-digits-dataset#Sign-language-digits-dataset.zip
X = np.load('./dataset/X.npy')
y = np.load('./dataset/Y.npy')
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

# Add 4 axis representing grey scale
X = X[:,:,:,np.newaxis]
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

# datagen = ImageDataGenerator(
#     rotation_range=16,
#     width_shift_range=0.12,
#     height_shift_range=0.12,
#     zoom_range=0.12
#     )

# datagen.fit(X)

# print(X.shape)
# print(len(X[0].shape)) # image is in grey scale because is less than 3

# print(y) # [ [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.], ..., [] ]
# print(y[0]) # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# print(len(y)) # 2062

# print(len(X[0])) # 64 
# print(len(X[0][0])) # 64 
# print(X[0][0]) # 0 - 1
# print(len(X)) # 2062

# Total images: 2062
# Image size: 64x64
# Labels/Classes:  10

# Show grey image
# plt.imshow(X[0], cmap="gray_r") # cmap="gray_r" or cmap="gray"
# plt.show()

# Normalization image
X = X / 255.0
# print(X[0])
# plt.imshow(X[0], cmap="gray")
# plt.show()


model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = (64, 64, 1))) # input_shape = X.shape[1:]
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10)) 
model.add(Activation("softmax"))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          epochs=20,
          validation_split=0.1,
          batch_size=32)

score, acc = model.evaluate(X, y, verbose=0)
print('\n')
print('Test score:', score)
print('Test accuracy:', acc)

# score = model.evaluate(x_test, y_test, batch_size=128)


# TODO: 
# - generate new images from dataset
#  - ranomize data
#     - merge X and Y
#     - split to tests?
# - add tensorboard
# - auto generate new models
# - save the best model
# - predict photos
# - validate?
