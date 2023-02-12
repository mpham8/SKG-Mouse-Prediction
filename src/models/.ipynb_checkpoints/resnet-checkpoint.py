import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class ResNet:
  
  def __init__(self, height, width, channels, classes):
    self.height = height
    self.width = width
    self.channels = channels
    self.classes = classes

    #CONFIGURATIONS
    self.batch_size = 128
    self.verbose = 1
    self.n = 3
    self.feature_maps = 16
    self.shortcut_type = 'identity'
    
    self.max_iterations = 64000
    self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.boundaries = [32000, 48000]
    self.learning_rates = [0.1, 0.01, 0.001]
    self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries, self.values)

    self.initializer = tf.keras.initializers.HeNormal()
    self.optimizer_momentum = 0.9
    self.optimizer_additional_metrics = ["accuracy"]
    self.optimizer = tf.keras.optimizersSGD(learning_rate=self.lr_schedule, momentum=self.optimizer_momentum)

    # Load Tensorboard callback
    tensorboard = tf.keras.callbacks.TensorBoard(
	  log_dir=os.path.join(os.getcwd(), "logs"),
	  histogram_freq=1,
	  write_images=True)

	  # Save a model checkpoint after every epoch
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
		os.path.join(os.getcwd(), 
    "model_checkpoint"),
		save_freq="epoch")

	  # Add callbacks to list
    self.callbacks = [tensorboard, checkpoint]
  
  # def residual_block(
  
  def train(self, x_train, y_train, x_test, y_test):
    self.n_training = len(x_train)
    self.n_test = len(x_test)
    self.steps_per_epoch = tf.math.floor(self.n_training/self.batch_size)
    self.val_steps_per_epoch = tf.math.floor(self.n_test/self.batch_size)
    self.epochs = tf.cast(tf.math.floor(self.max_iterations / steps_per_epoch), dtype=tf.int64)
    return
  
  def predict(self, x_test):
    self.n_training = len(x_test)
    return
  

#(input_train, _), (_, _) = load_dataset()



def data():
  #load dataset
  (input_train, target_train), (input_test, target_test) = cifar10.load_data()

  #one hot encoding
  target_train = tf.keras.utils.to_categorical(target_train, 2)
  target_test = tf.keras.utils.to_categorical(target_test, 2)

  # Data generator for training data
  train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
	validation_split = 0.1,
	horizontal_flip = True,
	rescale = 1./255,
	preprocessing_function = tensorflow.keras.applications.resnet50.preprocess_input
	)

  # Generate training and validation batches
  train_batches = train_generator.flow(input_train, target_train, batch_size=128, subset="training")
  validation_batches = train_generator.flow(input_train, target_train, batch_size=128, subset="validation")

	# Data generator for testing data
  test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
		preprocessing_function = tensorflow.keras.applications.resnet50.preprocess_input, 
    rescale = 1./255)

	# Generate test batches
  test_batches = test_generator.flow(input_test, target_test, batch_size=128)

  return train_batches, validation_batches, test_batches
  
data()

#model = ResNet(32, 32, 3, 10)