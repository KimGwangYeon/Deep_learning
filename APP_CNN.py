
#################### The aim of the lesson ###################
# 1. learning method of preventing over fitting
# 2. Build network model for optimized CNN
# 3. Explain CIFAR-10
##############################################################

# For preventing over fitting - Batch Normalization
# make output having a normal distribution (ave : 0, dist = 1)

# For preventing over fitting - Dropout
# make neurons disconnected

# Optimizing CNN - Multi Conv Layer + Batch Normalization + Dropout

# EXAMPLE

# DATA LOAD
import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels)=mnist.load_data() # data shape of mnist
print('shape of Train(Test) :', train_images.shape)

# reshape data
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
print('shape of Train(Test) :', train_images.shape)

# scaling data
train_images = train_images.astype('float32')/255 # for dividing 255,
test_images = test_images.astype('float32')/255

# encoding
train_images = to_categorical(train_labels) # one_hot encoding
test_labels = to_categorical(test_labels)

# BUILD NETWORK MODEL
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.layers.normalization import BatchNormalization

model = Sequential() # I'm gonna make Multi Conv Layer

# First Conv Layer
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_size=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3 ,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

# Second Conv Layer
model.add(Conv2D(64, kernel_size=(3, 3),  padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

# Third Conv Layer
model.add(Conv2D(128, kernel_size=(3, 3),  padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3),  padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))

# Making data Flatt for inputting to neural network
model.add(Flatten())

# neural network
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# OPTIMIZER - ADAM

from keras.optimizers import Adam
optimizer = Adam(lr=0.001)

# setting my model
model.comple(optimizer=optimizer,
             loss='categorical_crossentropy', # set loss crossentropy(classification)
             metrics=['accuracy'])

model.summary()

# TRAINING MODEL
model.fit(train_images, train_labels, epoch=5, batch_size=200)
# sequence : train_input, train_label, epoch, batch_size

# EVALUATE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels) # Memorize It!!

# print(test_acc)