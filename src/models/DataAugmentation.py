import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('../../data/UCr/abundance.tsv', index_col=0, sep='\t', header=None)
to_drop = data.loc[(data < 0.1).all(axis=1)]
data = data.drop(to_drop.index)
labels = np.genfromtxt('../../data/UCr/labels.txt', dtype=np.str_, delimiter=',')

data_array = np.array(data).T

class_0 = data_array[labels.astype(int) == 0]
class_1 = data_array[labels.astype(int) == 1]

print(class_0.shape)
print(class_1.shape)



# Size of latent vector
latent_dim = 10
epochs = 1000
# Reduce Batch Size if GPU memory overflow
# Better Use Google Colab
batch_size = 2000
# Save model after every
saving_rate = 100

# Random Seed for Shuffling Data
buffer_size = 5000

# defining optimizer for Models
gen_optimizer = tf.keras.optimizers.Adam(0.0001)
disc_optimizer = tf.keras.optimizers.Adam(0.0001)

# Shuffle & Batch Data
#train_dataset = train_data.map(normalize).shuffle(buffer_size , reshuffle_each_iteration=True).batch(batch_size)
train_dataset = class_0


# Define Loss Function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu )
        self.dense_2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu )
        self.dense_3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense_4 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense_4 = tf.keras.layers.Dense(22)
        #self.reshape = tf.keras.layers.Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.reshape(x)


class GAN(tf.keras.Model):
    # define the models
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
    
    # Define the compiler
    def compile(self, disc_optimizer, gen_optimizer, loss_fn, generator_loss, discriminator_loss):
        super(GAN, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.loss_fn = loss_fn

        
    # @tf.function: The below function is completely Tensor Code
    # Good for optimization
    @tf.function
    # Modify Train step for GAN
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # Define the loss function
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generator(noise)
            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)
            
            gen_loss = self.generator_loss(self.loss_fn, fake_output)
            disc_loss = self.discriminator_loss(self.loss_fn, real_output, fake_output)

        # Calculate Gradient
        grad_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        grad_gen = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Optimization Step: Update Weights & Learning Rate
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        
        return {"Gen Loss ": gen_loss,"Disc Loss" : disc_loss}


# Save Image sample from Generator
# epoch: Number of epoch trained
# noise: latent vector
def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

class Training_Callback(tf.keras.callbacks.Callback):
    # Constructor
    def __init__(self, latent_dim, saving_rate):
        super(Training_Callback, self).__init__()
        self.latent_dim = latent_dim
        self.saving_rate = saving_rate
        
    # Save Image sample from Generator
    def save_imgs(self, epoch):
        # Number of images = 16
        seed = tf.random.normal([16, self.latent_dim])
        gen_imgs = self.model.generator(seed)
        
        fig = plt.figure(figsize=(4, 4))

        for i in range(gen_imgs.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        fig.savefig("images/mnist_%d.png" % epoch)
    
    # Called after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # Save image after 50 epochs
        if epoch % 50 == 0:
            self.save_imgs(epoch)
        
        # Save Model every 100 epoch
        if epoch > 0 and epoch % self.saving_rate == 0:
            save_dir = "./models/model_epoch_" + str(epoch)
            
            # Create Directory to save file
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save weights
            self.model.discriminator.save_weights(save_dir + '/discriminator_%d' % epoch)
            self.model.generator.save_weights(save_dir + '/generator_%d' % epoch)


def data():
    # data = pd.read_csv('../../data/UCr/abundance.tsv', index_col=0, sep='\t', header=None)
    # to_drop = data.loc[(data < 0.1).all(axis=1)]
    # data = data.drop(to_drop.index)
    # labels = np.genfromtxt('../../data/UCr/labels.txt', dtype=np.str_, delimiter=',')

    # data_array = np.array(data).T

    # class_0 = data_array[labels.astype(int) == 0]
    # class_1 = data_array[labels.astype(int) == 1]

    # print(class_0.shape)
    # print(class_1.shape)
    
    # k = 5
    
    # #load data set
    # (input, target), (_, _) = mnist.load_data()
    # target = tf.keras.utils.to_categorical(target, 10, dtype='uint8')
    # #t2 = tf.keras.utils.to_categorical(t2, 10, dtype='uint8')


    # x_train = input
    # y_train = target
    # # x_test = i2
    # # y_test = t2


    # #one hot encoding
    
    
    # # model = ResNet(height = 28, width = 28, channels = 1, classes = 10)
    # # model.init_model()
    # # model.train(x_train, y_train)
    # # y_pred = model.predict(x_test)
    # # model.evaluate(y_test, y_pred)
    # # print(model.model.summary())

    # print(input.shape)
    # print(target.shape)
    disc = Discriminator()
    gen = Generator()

    # Create compile model
    gan = GAN(discriminator=disc, generator=gen, latent_dim=latent_dim)
    gan.compile(
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        loss_fn=cross_entropy,
        generator_loss = generator_loss,
        discriminator_loss = discriminator_loss
    )

    # Start training
    training_callback = Training_Callback(10, saving_rate)
    gan.fit(
        train_dataset, 
        epochs=epochs,
        callbacks=[training_callback]
    )

# # Save Model
# disc.save_weights('./models/discriminator')
# gen.save_weights('./models/generator')
    
    return

data()