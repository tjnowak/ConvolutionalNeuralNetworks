#! /N/u/tjnowak/BigRed2/aconda2/bin/python

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from __future__ import print_function
import math
from PIL import ImageFile              # stops PIL from thowing 'image
ImageFile.LOAD_TRUNCATED_IMAGES = True # 'file is truncated' error


img_width, img_height = 302, 302 # images resized during training

weights_path = 'OwnModelFinalWeights.h5'
test_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/final_test'
nb_test_samples = 2389
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Rebuild model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))

# Load trained model weights
model.load_weights(weights_path)

# Configure compiler
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Configure test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False)

# Run test data through model once and save loss & accuracy results
evaluate_size_test = int(math.ceil(nb_test_samples / float(batch_size)))  # evaluate_generator can't calculate                                                                  
results = model.evaluate_generator(test_generator, evaluate_size_test)    # this internally if doesn't divide evenly
f=open('results.txt', 'w')
print('Test Loss: ', results[0], file=f)      # save loss
print('Test Accuracy: ', results[1], file=f)  # save accuracy

