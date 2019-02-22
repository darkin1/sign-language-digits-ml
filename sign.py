# import tensorflow as tf
import numpy as np
import sys
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from numpy import array

### Load dataset
#@link: https://www.kaggle.com/ardamavi/sign-language-digits-dataset#Sign-language-digits-dataset.zip
X = np.load('./dataset_fixed/X.npy')
y = np.load('./dataset_fixed/Y.npy')
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

### Add 4 axis representing grey scale
X = X[:,:,:,np.newaxis]
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

### Randomize dataset
shuffle_index = np.random.permutation(2062)
X, y = X[shuffle_index], y[shuffle_index]

### Check labels
# label = 2
# print("Number: %s" % y[label])
# plt.imshow(np.squeeze(X[label]), cmap='gray')
# plt.show()
# sys.exit()

### Split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

print(f'X_train: {len(X_train)}; y_train: {len(y_train)}')
print(f'X_test: {len(X_test)}; y_test: {len(y_test)}')


### Generate new training images 
datagen = ImageDataGenerator(
    rotation_range=16,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    # validation_split=0.25
)
datagen.fit(X_train)


### Show grey image
# plt.imshow(X[0], cmap="gray_r") # cmap="gray_r" or cmap="gray"
# plt.show()


### Path to model logs
NAME = "sign-10-40-epochs-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs1/{}'.format(NAME))


### Create Model
def create_model():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = (64, 64, 1))) # input_shape = X.shape[1:]
    model.add(Activation("relu")) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu")) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu")) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(10)) 
    model.add(Activation("softmax"))

    model.summary()

    ### Run model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adadelta(),
                metrics=['accuracy'])
    return model

model = create_model()

### Train and validate model
model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=128,
                    epochs=5,
                    validation_data=(X_test, y_test),
                    callbacks=[tensorboard]
                )

### Simpler validation
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Val Loss: {val_loss}; Val Accuracy: {val_acc}" )


### Predict answer
key = 1
pX = X[key][np.newaxis,:,:,:] # that same as: array([X[1]])

predicted = model.predict(pX)
print(predicted)

# predicted_proba = model.predict_proba(pX)
# print(predicted_proba)

predicted_class_number = model.predict_classes(pX)
print(predicted_class_number)

print("Classes probability=%s, The best predicted number=%s, Y=%s" % (predicted, predicted_class_number, y[key]))

print(np.round(predicted_class_number, 1))


# Show predicted image
plt.imshow(np.squeeze(X[key]), cmap="gray")
plt.show()
# sys.exit()