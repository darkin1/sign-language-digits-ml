# import tensorflow as tf
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/ardamavi/sign-language-digits-dataset#Sign-language-digits-dataset.zip
X = np.load('./dataset/X.npy')
y = np.load('./dataset/Y.npy')
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

# Add 4 axis representing grey scale
X = X[:,:,:,np.newaxis]
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))
# print(y[0])

# Normalization image
# X /= 255.0

shuffle_index = np.random.permutation(2062)
X, y = X[shuffle_index], y[shuffle_index]
# print(y[0])


### Split test and train data

train_length = round(len(X)*0.75)
test_length = round(len(X)*0.25)
# print(f'train_length: {train_length}; test_length: {test_length}')

X_train = X[:train_length]
X_test = X[-test_length:]

y_train = y[:train_length]
y_test = y[-test_length:]

## Inny sposob na dzielenie datasetu
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

print(f'X_train: {len(X_train)}; y_train: {len(y_train)}')
print(f'X_test: {len(X_test)}; y_test: {len(y_test)}')

# Generate new images
datagen = ImageDataGenerator(
    rotation_range=16,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12
)

datagen.fit(X_train)
# X_train = np.array(X_train).reshape(-1, 64, 64, 1)
# X_test = np.array(X_test).reshape(-1, 64, 64, 1)
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32, epochs=epochs)
# sys.exit()

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
             optimizer=optimizers.Adadelta(),
             metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=10,
          batch_size=32
          )

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('\n')
print('Val loss:', val_loss)
print('Val accuracy:', val_acc)
# Jesli są duże rozbierzności (zbyt blisko/daleko - zbyt duża delta) pomiędzy `loss/acc` z ostatniej generacji a `val_loss/val_acc` z model.evaluate
# to najprawdopodobniej model jest przetrenowany (overfitting)
# Gdy dochodzi do przetrenowania to model zamiast znajdować zależności w danych, zapamiętuje wszystkie przykłady
# Wtedy też warto zmniejszyć ilość epok??
# Powinniśmy się spodziewać że (in sample accuracy), czyli dane z ostatniej generacji epok będą lekko mniejsze niż te z val_loss, val_acc

# acc/loss - in sample accuracy/loss  - dane na których trenujemy (epoki)
# val_acc/val_loss - out of sample accuracy/loss - dane na ktorych testujemy (chyba??)

# print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score,acc))

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
