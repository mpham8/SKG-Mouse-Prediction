"""
  Winter 2022
  Michael Pham
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import datetime


class ResNet:
    def __init__(self, height, width, channels, classes):
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes

        #CONFIGURATIONS
        self.epochs = 12  
        self.batch_size = 2
        self.optimizer = tf.keras.optimizers.Adam()
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #self.loss = 'binary_crossentropy'
        #self.loss = 'categorical_crossentropy'

        # self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/path/to/save/model.h5',
        #                                         monitor='acc',
        #                                         save_best_only=True,
        #                                         save_weights_only=False)
        self.log_dir = "../tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        self.initial_learning_rate = 0.1
        self.boundaries = [1000, 2000, 4000]
        self.values = [self.initial_learning_rate, 0.01, 0.001, 0.0001]



    def identity_block(self, x, filters, size, names):
        """
        The identity block which performs the following operations:
        skips x
        applies 3x3 convulution, [filters] filters, stride of 1, same padding to x
        applies batch normalization to x
        applies ReLu activation to x
        applies 3x3 convulution, [filters] filters, stride of 1, same padding to x
        applies batch normalization to x
        adds skipped x to x
        applies ReLu activation to x

        Attributes:
          x (tensor): Tensorflow tensor
          filters (int): number of filters
          names (str): name of layer

        Returns:
          x(tensor): Tensorflow tensor
        """
        
        x_skip = x
        
        #conv 1
        x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (size[0],size[1]), strides = (1,1), padding = 'same', name = 'res' + names + '-conv1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        x = tf.keras.layers.BatchNormalization(axis = 3, name = 'res' + names + '-batchnorm1')(x)
        x = tf.keras.layers.Activation('relu')(x)

        #conv 2
        x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (size[2],size[3]), strides = (1,1), padding = 'same', name = 'res' + names + '-conv2', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        x = tf.keras.layers.BatchNormalization(axis = 3, name = 'res' + names + '-batchnorm2')(x)

        #skip connection
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def convolutional_block(self, x, filters, names):
        """
        The convolutional block which performs the following operations:
        skips x
        applies 3x3 convulution, [filters] filters, stride of 1, same padding to x
        applies batch normalization to x
        applies ReLu activation to x
        applies 3x3 convulution, [filters] filters, stride of 1, same padding to x
        applies batch normalization to x
        adds skipped x to x
        applies ReLu activation to x

        Attributes:
          x (tensor): Tensorflow Tensor
          filters (int): number of filters
          names (str): name of layer

        Returns:
          x(tensor): Tensorflow Tensor
        """
        
        #skip
        x_skip = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1), strides = (1,2), padding = 'same', name = 'res' + names + '-convi', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)

        #conv 1
        x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,2), padding = 'same', name = 'res' + names + '-conv1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        x = tf.keras.layers.BatchNormalization(name = 'res' + names + '-batchnorm1')(x)
        x = tf.keras.layers.Activation('relu')(x)

        #conv 2
        x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = 'same', name = 'res' + names + '-conv2', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        x = tf.keras.layers.BatchNormalization(name = 'res' + names + '-batchnorm2')(x)

        #skip connection
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)


        return x

    def init_model(self):
        """
        The model
        """

        #set input layer

        input = tf.keras.layers.Input((self.height, self.width, self.channels))

        #Conv 1
        #x = tf.keras.layers.Conv2D(64, (3, 6), strides = (1, 2), name = 'conv1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(input)
        #x = tf.keras.layers.BatchNormalization(name = 'bn1')(x)
        #x = tf.keras.layers.Activation('relu')(x)

        #Residual Block 1
        #x = tf.keras.layers.Conv2D(32, (4, 24), strides = (3, 3), name = 'convmid', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(input)
        #x = tf.keras.layers.BatchNormalization(name = 'bn2')(x)
        #x = tf.keras.layers.Activation('relu')(x)
        x = self.identity_block(input, 64, [2,4,2,4], '2a')
        #x = tf.keras.layers.Conv2D(64, (3, 3), strides = (2,2), padding = 'same', name = 'convmid2', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(input)
        x = self.identity_block(x, 64, [2,4,2,4], '2b')
        #x = tf.keras.layers.Conv2D(64, (3, 10), strides = (1,1), padding = 'same', name = 'convmid3', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(input)
        #x = self.identity_block(x, 64, [2,4,2,4], '2c')

        #x = self.identity_block(x, 32, [3,10,3,10], '2c')
        #x = self.identity_block(x, 64, '2d')
        # x = self.identity_block(x, 64, '2e')
        # x = self.identity_block(x, 64, '2f')
        # x = self.identity_block(x, 64, '2g')
        # x = self.identity_block(x, 64, '2h')

        #Residual Block 2 w/ Projection
        #x = self.convolutional_block(x, 128, '2b')
        
        
        x = tf.keras.layers.Conv2D(1, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        
        
        #flatten
        x = tf.keras.layers.Flatten()(x)
        
        #fc
        x = tf.keras.layers.Dense(100, activation = 'sigmoid', name = 'fc', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 
        
        #x = tf.keras.layers.Dropout(rate=0.0)(x)

        
        #x = tf.keras.layers.Dense(32, activation = 'sigmoid', name = 'fc2', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 
        
        #x = tf.keras.layers.Dense(25, activation = 'relu', name = 'fc3', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 

        #dropout
        #x = tf.keras.layers.Dropout(rate=0.1)(x)

        #softmax
        x = tf.keras.layers.Dense(2, activation = 'softmax', name = 'softmax', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)
        
        #fc
        self.model = tf.keras.models.Model(inputs = input, outputs = x, name='res')

        #self.model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(self.height, self.width, 1)))

        
        return

    def train(self, x_train, y_train, x_test, y_test, path_weights, use_weights):
    #def train(self, x_train, y_train, x_test, y_test):

        """
        Compiles, then trains the model
    
        Attributes:
          x_train (np,matrix): Matrix of indepedent variables from the training set
          y_train (np.matrix): Matrix of labels from the training set
        """

        self.model.compile(self.optimizer, self.loss, metrics = ['accuracy'])

        if use_weights == True:
          self.model.load_weights("../data/" + path_weights + "/model_weights.h5")

        def schedule(epoch):
          # if epoch < 65:
          #   return 0.001
          # elif epoch >= 65 & epoch < 80:
          #   return 0.0005
          # elif epoch >= 125:
          #   return 0.0001
          if epoch < 100:
            return 0.001
          elif epoch >= 100 & epoch < 125:
            return 0.0005
          elif epoch >= 150:
            return 0.0001

        #self.model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size, validation_data=(x_test, y_test), callbacks=[self.callback, tf.keras.callbacks.LearningRateScheduler(schedule)])
        self.model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size, validation_data=(x_test, y_test), callbacks=[self.callback])

        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        return

    def predict(self, x_test):
        """
        Predicts labels
    
        Attributes:
          x_test (np.matrix): Matrix of indepedent variables from the test set
        
        Returns:
          (float): Matrix of predicted continous probabilities of labels
        """
        
        return self.model.predict(x_test)

    def evaluate(self, y_test, y_pred):
        """
        Computes evaluation metrics
    
        Attributes:
          y_test (np.matrix): Matrix of one hot encoded labels from the test set
          y_pred (np.matrix): Matrix of predicted continous probabilities of labels
        
        Returns:
          auc_roc (float): Area under the curve (AUC PR) Score
          auc_pr (float): Area under the precision recall curve (AUC ROC) Score
          f1 (float): F1 Score 
          mcc (float): Matthews correlation coefficient (MCC) Score
        """

        # Compute the AUC of the ROC curve
        auc_roc = roc_auc_score(y_test, y_pred)

        # Compute the AUC of the PR curve
        auc_pr = average_precision_score(y_test, y_pred)

        # Compute the F1 score
        y_pred = np.argmax(y_pred, axis=1)
        y_pred_multiclass = tf.keras.utils.to_categorical(y_pred, self.classes, dtype='uint8')
        f1 = f1_score(y_test, y_pred_multiclass, average='weighted')

        # Compute the MCC
        y_test = np.argmax(y_test, axis=1)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Print the evaluation results
        print("auc_roc:", auc_roc)
        print("auc_pr:", auc_pr)
        print("f1_score:", f1)
        print("mcc:", mcc)

        #tf.keras.backend.clear_session()
        return auc_roc, auc_pr, f1, mcc

    def destroy(self):
      tf.keras.backend.clear_session()
      return