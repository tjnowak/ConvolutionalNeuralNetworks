#! /N/u/tjnowak/BigRed2/aconda2/bin/python

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.callbacks import CSVLogger


img_width, img_height = 302, 302 # images resized during training

train_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/train'
validation_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/validation'
nb_train_samples = 7530
nb_validation_samples = 1989
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Build convolutional base
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

# Build fully-connected classifier
model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))

# Configure compiler
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Configure train generator. Augment images to provide
# additional training examples.
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # rescale pixel values between 0-1
    shear_range=0.2,  # skew images
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Configure test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Train model & save weights
csv_logger = CSVLogger('OwnModelFinal.csv') # log loss & accuracy during training
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[csv_logger])
model.save_weights('OwnModelFinalWeights.h5')
