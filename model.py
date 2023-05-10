import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten,Dropout,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import warnings
warnings.filterwarnings("ignore")

path = f'Mask/'
train_path = path + 'train/'
valid_path = path + 'valid/'
test_path = path + 'test/'

INPUT_SHAPE = (460,460,3)
NUM_CLASSES=4
train_datagen = ImageDataGenerator(
    dtype='float32',
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

val_datagen = ImageDataGenerator(
    dtype='float32',
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    dtype='float32',
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(460,460),
    batch_size=32,
    class_mode='categorical',  
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(460,460),
    batch_size=32,
    class_mode='categorical',
)

validation_generator = val_datagen.flow_from_directory(
    valid_path,
    target_size=(460,460),
    batch_size=32,
    class_mode='categorical',
)

base_model = ResNet50(include_top=False,pooling='av',weights='imagenet',input_shape=(INPUT_SHAPE))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES,activation='softmax'))
plot_model(model,to_file="resnet50.png",show_shapes=True,show_layer_names=True)

optimizer = tf.keras.optimizers.Adam(learning_rate= 0.00001)

model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    filepath='model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(
    patience=10,
    verbose=1
)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

result = model.evaluate(test_generator)
