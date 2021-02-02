import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class dcgan():
  def __init__(self, dDepth=1, gDepth=1, batch_size=128, filters=1):
    self.dDepth = dDepth
    self.gDepth = gDepth
    self.batch_size = batch_size
    self.filters = filters
    self.D = self.discriminator()
    self.G = self.generator()
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
    #self.model = tf.keras.Sequential([self.D,self.G])
  
  def discriminator_loss(self,real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss

  def generator_loss(self,fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

  def discriminator(self):
    model = tf.keras.Sequential()
    model.add(layers.UpSampling2D(size=(2,2)))
    model.add(layers.Conv2D(64*self.filters,(5,5),strides=(2,2),padding='same',input_shape=[56,56,1]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64*self.filters,(5,5),strides=(2,2),padding='same',input_shape=[28,28,1]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128*self.filters,(5,5),strides=(2,2),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(4*self.filters))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    #print(model.summary())
    return model

  def generator(self):
    model = tf.keras.Sequential()
    model.add(layers.Dense(units=1024,input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(7*7*256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7,7,256)))

    model.add(layers.Conv2DTranspose(128*self.filters,(5,5),strides=(1,1),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64*self.filters,(5,5),strides=(2,2),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))

    #print(model.summary())
    return model

  @tf.function
  def train_step(self,images,noise_dim):
    noise = tf.random.normal([self.batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.G(noise, training=True)

      real_output = self.D(images, training=True)
      fake_output = self.D(generated_images, training=True)

      gen_loss = self.generator_loss(fake_output)
      total_loss, disc_loss, gen_loss_inv = self.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(total_loss, self.D.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    return gen_loss, disc_loss

  def train(self,x_train,y_train=None,epochs=10,noise_dim=100):
    mean_gen_loss = []
    mean_disc_loss = []

    for epoch in range(epochs):
      start = time.time()
      batch_gen_loss = []
      batch_disc_loss = []

      for image_batch in x_train:
        gen_loss, disc_loss = self.train_step(image_batch,noise_dim)
        batch_gen_loss.append(gen_loss.numpy())
        batch_disc_loss.append(disc_loss.numpy())

      mean_gen_loss.append(np.mean(batch_gen_loss))
      mean_disc_loss.append(np.mean(batch_disc_loss))

      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      self.generate_and_save_images(self.G, epoch + 1, noise_dim)

      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
      print ('Generator loss: {}'.format(mean_gen_loss[epoch]))
      print ('Discriminator loss: {}'.format(mean_disc_loss[epoch]))

    # Generate after the final epoch
    display.clear_output(wait=True)
    self.generate_and_save_images(self.G, epochs, noise_dim)
    
    return mean_gen_loss, mean_disc_loss

  def generate_and_save_images(self,model,epoch,noise_dim):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    test_input = tf.random.normal([10, noise_dim])

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

