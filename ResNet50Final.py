#! /N/u/tjnowak/BigRed2/aconda2/bin/python

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import applications
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical 
from keras.callbacks import CSVLogger
import math 


img_width, img_height = 224, 224 # images resized during training

top_model_weights_path = 'ResNet50FinalWeights.h5'
csv_name = 'ResNet50Final.csv'

train_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/train'
validation_data_dir = '/N/u/tjnowak/BigRed2/leaves/processed_data/data/validation'

nb_train_samples = 7530
nb_validation_samples = 1989
num_classes = 11
epochs = 350
batch_size = 16

# Load the ResNet50 model, run the leaf data through it once, and save the features
def save_bottleneck_features():
    # Build the ResNet50 network
    model = applications.resnet50.ResNet50(include_top=False, weights='imagenet') # don't include top classifier

    # Configure train generator
    datagen = ImageDataGenerator(rescale=1. / 255) # rescale pixel values between 0-1
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,  # labels not needed for predict_generator
        shuffle=False)

    # Run train data through model and save features
    predict_size_train = int(math.ceil(nb_train_samples / float(batch_size)))  # predict_generator can't calculate                                                                  
    bottleneck_features_train = model.predict_generator(                       # this internally if doesn't divide evenly
        generator, predict_size_train)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    # Configure validation generator
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # Run validation data through model and save features
    predict_size_validation = int(math.ceil(nb_validation_samples / float(batch_size)))                           
    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


# Use the saved ResNet50 features to train a new top classifier model and save the classifier weights
def train_top_model():
    # Load the saved bottleneck features  
    train_data = np.load(open('bottleneck_features_train.npy'))
    validation_data = np.load(open('bottleneck_features_validation.npy'))

    # Get the class labels for the train data using a generator
    datagen_top = ImageDataGenerator(rescale=1./255)  
    generator_top = datagen_top.flow_from_directory(  
        train_data_dir,  
        target_size=(img_width, img_height),  
        batch_size=batch_size,  
        class_mode='categorical', 
        shuffle=False)
    train_labels = to_categorical(generator_top.classes, num_classes=num_classes)  # convert class vector to
                                                                                   # class matrix
    # Get the class labels for the validation data using a generator
    generator_top = datagen_top.flow_from_directory(  
        validation_data_dir,  
        target_size=(img_width, img_height),  
        batch_size=batch_size,  
        class_mode='categorical',  
        shuffle=False)
    validation_labels = to_categorical(generator_top.classes, num_classes=num_classes)  

    # Create the top classification model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256))
    model.add(PReLU())
    model.add(Dense(256))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Configure learning process
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy', 
		          metrics=['accuracy'])

    # Train top model with bottleneck features as input and save weights
    csv_logger = CSVLogger(csv_name)  # store loss & accuracy for each epoch
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[csv_logger])
    model.save_weights(top_model_weights_path)


#save_bottleneck_features()
train_top_model()