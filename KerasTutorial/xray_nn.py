import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from scipy import misc
from sklearn.model_selection import train_test_split


def load_images(images_path, path_to_labels):
    path_to_images = images_path
    labels_file = path_to_labels
    images_labels = {}
    with open(labels_file, 'r') as f:
        dict_labels = dict([line.strip().split() for line in f.readlines()])

    files = os.listdir(path_to_images)

    files = filter(lambda files: not files.startswith('.'), files)

    # Create structure for holding images
    images = np.zeros((4999, 3, 64, 64), dtype=np.uint8)
    labels = np.zeros(4999, dtype=np.uint8)
    for fid, file in enumerate(files):
        if fid % 1000 == 0:
            print(fid)
        image = misc.imread(path_to_images + '/' + file)
        if (image.shape == (64, 64, 4)):
            image = image[:, :, :3]
        if image.shape == (64, 64):
            print(file)
        images[fid] = image.T
        labels[fid] = int(dict_labels[file])
    return images, labels

dataset, labels = load_images("/home/oleksandr/CS@UCU/Course3/X-rayProject/resized_images/", "/home/oleksandr/CS@UCU/Course3/X-rayProject/images_labels.txt")
train_images, test_images, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.33, random_state=42)

print('Training data shape : ', train_images.shape, train_labels.shape)
print('Testing data shape : ', test_images.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

print(train_data.shape)

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation='softmax'))

opt = SGD(lr=0.01)
model_reg.compile(loss = "categorical_crossentropy", optimizer = opt,  metrics=['accuracy'])

history = model_reg.fit(train_data, train_labels_one_hot, batch_size=64, epochs=50, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model_reg.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()