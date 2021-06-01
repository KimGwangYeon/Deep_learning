
# CNN

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

model = Sequential()

# extracting feature maps from input image (# : 32)
model.add(Conv2D(32, kernel_size=(3, 3), input_size=(28 ,28, 1), activation='relu'))

# extracting features from feature maps
model.add(MaxPooling2D(pool_size=2))

# flatten features for input Deep-learning
model.add(Flatten())

# building neural network
model.add(Dense(128, activation='relu'))
model.add(Dense(19, activation='softmax'))

# model summary
model.summary()

# possible to adding more block(conv, acti, max) = Multi Convolution Layer
# It's up to character of input data