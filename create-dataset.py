import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import traceback

IMG_SIZE = 64
DATADIR = "./dataset_raw"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# for category in CATEGORIES:    
#     path = os.path.join(DATADIR, category) #path to cats or dogs dir ex. ./../datasets/kagglecatsanddogs/PetImages/Dog
#     for img in os.listdir(path): # ex. img == 7949.jpg
        
#         # konwerujemy zdjęcia na szare
#         # poniewaz zajmuje mniej miejsca
#         # poniewaz kolor nie definiuje czy jest to kot czy pies
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
#         # plt.imshow(img_array, cmap="gray")
#         # plt.show()
#         # break
#     # break
#     print(len(os.listdir(path)))

# print(img_array)
# print("\n Kształt tablicy ze zdjęciami: \n")
# print(img_array.shape)

X = []
y = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) # 0 - Dog; 1 - Cat
        # print(class_num)
        for img in os.listdir(path):
            try: #dodajemy try ponieważ niektóre pliki są zepsute
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # konwertujemy na szare zdjęcia
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # skalujemy zdjęcia do tego same rozmiaru
                # X.append([new_array, class_num]) # wrzucamy skonwertowane zdjęcia do nowej tablicy
                X.append(new_array) # wrzucamy skonwertowane zdjęcia do nowej tablicy
                oneHot = tensorflow.keras.utils.to_categorical(category, 10) # convert ot on-hot
                y.append(oneHot)
            except Exception as e:
                traceback.print_exc()
        
create_training_data()

X = 1-np.array(X).astype('float32')/255.
X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE)  # X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)

np.save('./dataset_fixed/X.npy', X)
np.save('./dataset_fixed/Y.npy', y)

key = 1852
print(len(X))
print(X.shape)
# print(X[0])
# print(y)

### Print the label converted back to a number
# label = y[key].argmax(axis=0)
# print(label)

### Show image
# plt.imshow(np.squeeze(X[key]), cmap="gray")
# plt.show()