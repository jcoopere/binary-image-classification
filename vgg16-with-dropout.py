'''
!pip3 install -U numpy tensorflow-gpu keras pillow h5py
'''

import os
import sys
import time
import glob
import matplotlib.pyplot as plt
from keras import layers, models, optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator

# Globals
image_dimensions = 224
num_channels = 3

def count_images_in_dir(path):
  # This code assumes .tiff image format...to accept a different format, modify the following line.
  return len(glob.glob(os.path.join(path, '*', '*.tiff')))

def smooth_curve(points, factor=0.8):
  smoothed_points = []
  
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  
  return smoothed_points
      
def train_cnn(corpus_path):
  train_dir = os.path.join(corpus_path, 'train')
  validate_dir = os.path.join(corpus_path, 'validate')
  test_dir = os.path.join(corpus_path, 'test')

  training_image_count = count_images_in_dir(train_dir)
  validation_image_count = count_images_in_dir(validate_dir)
  testing_image_count = count_images_in_dir(test_dir)

  print('INFO: {} images will be used for training ({})'.format(training_image_count, train_dir))
  print('INFO: {} images will be used for validation ({})'.format(validation_image_count, validate_dir))
  print('INFO: {} images will be used for testing ({})'.format(testing_image_count, test_dir))

  labels = os.listdir(train_dir)
  num_labels = len(labels)
  print("INFO: Training set contains the following {} labels...".format(num_labels))
  for label in labels:
    print('  - {}'.format(label))

  # Train
  K.clear_session()

  model = models.Sequential()
  
  model.add(layers.Conv2D(64, (3, 3), input_shape=(image_dimensions, image_dimensions, num_channels), padding='same', activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Flatten())
  model.add(layers.Dense(4096, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(4096, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_labels, activation='softmax'))

  model.summary()

  model.compile(optimizers.RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

  train_datagen = ImageDataGenerator(rescale=1./255)
  validate_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_dimensions, image_dimensions), batch_size=20, shuffle=True)
  validate_generator = validate_datagen.flow_from_directory(validate_dir, target_size=(image_dimensions, image_dimensions), batch_size=20, shuffle=True)

  for data_batch, labels_batch in train_generator:
  	print('INFO: Data batch shape: {}'.format(data_batch.shape))
  	print('INFO: Labels batch shape: {}'.format(labels_batch.shape))
  	break

  t_start = time.time()
    
  history = model.fit_generator(
  	train_generator,
  	steps_per_epoch=training_image_count // 20 + 1,
    epochs=300,
    validation_data=validate_generator,
    validation_steps=validation_image_count // 20 + 1)

  t_end = time.time()
  
  elapsed_time = t_end - t_start
  
  print('INFO: Training complete! Elapsed time: {}'.format(elapsed_time))
  
  print('INFO: Saving model as vgg16-classifier-model.h5')
  
  model.save('vgg16-classifier-model.h5')
  
  print('INFO: Plotting training & validation accuracy and loss...')
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.ylim(0.8, 1.0)
  plt.plot(epochs, acc, 'bo')
  plt.plot(epochs, smooth_curve(val_acc), 'b')
  plt.title('Training & Validation Accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'bo')
  plt.plot(epochs, smooth_curve(val_loss), 'b')
  plt.title('Training & Validation Loss')
  plt.show()

def test_cnn(corpus_path, model_path):
  from keras.models import load_model
  model = load_model(model_path)

  test_dir = os.path.join(corpus_path, 'test')

  testing_image_count = count_images_in_dir(test_dir)

  print('INFO: {} images will be used for testing'.format(testing_image_count))

  # Test
  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_dimensions, image_dimensions), batch_size=20)
  test_loss, test_acc = model.evaluate_generator(test_generator, testing_image_count // 20 + 1)

  print('INFO: Model accuracy on test data set is {}'.format(test_acc))
  
# Train, test, and save CNN
split_corpus_path = '/home/cdsw/split-sorted-images'
train_cnn(split_corpus_path)
test_cnn(split_corpus_path, '/home/cdsw/vgg16-classifier-model.h5')
