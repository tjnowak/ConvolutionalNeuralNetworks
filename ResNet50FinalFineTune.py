#! /N/u/tjnowak/BigRed2/aconda2/bin/python

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras import applications
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical 
from keras.callbacks import CSVLogger


img_width, img_height = 224, 224 # images resized during training

top_model_weights_path = 'ResNet50FinalWeights.h5'
new_weights_path = 'FineTunedResNet50FinalWeights.h5'
csv_name = 'FinedTunedResNet50Final.csv'

train_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/train'
validation_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/validation'

nb_train_samples = 7530
nb_validation_samples = 1989
num_classes = 11
epochs = 50
batch_size = 16

# Build the ResNet50 network
model = applications.resnet50.ResNet50(include_top=False, weights='imagenet') # don't include classifier
train_data = np.load(open('bottleneck_features_train.npy'))  # get shape of last ResNet50 
model.add(Flatten(input_shape=train_data.shape[1:]))         # feature maps

# Freeze all layers in the convolutional base except the last convolutional block
for layer in model.layers[:-15]: 
    layer.trainable = False

# Build the classifier
top_model = Sequential()
top_model.add(Dense(256))
top_model.add(PReLU())
top_model.add(Dense(256))
top_model.add(PReLU())
top_model.add(Dropout(0.5)) 
top_model.add(Dense(num_classes))  
top_model.add(Activation('softmax'))

# Load the classifier weights
top_model.load_weights(top_model_weights_path)

# Add the classifier to the convolutional base
model.add(top_model)

# Configure learning process
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), # Keras blog recommends SGD for fine-tuning
              loss='categorical_crossentropy',      # use low learning rate
              metrics=['accuracy'])

# Configure train generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # rescale pixel values between 0-1
    shear_range=0.2,  # skew images
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Configure validation generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Fine-tune the model and save the weights 
csv_logger = CSVLogger(csv_name)  # store loss & accuracy after each epoch
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[csv_logger])
model.save_weights(new_weights_path)