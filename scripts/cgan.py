import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.utils.np_utils import to_categorical   
import time
from IPython import display
import matplotlib.pyplot as plt
import inception_model, inception_score

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class cgan():
  def __init__(self, dDepth=1, gDepth=1, batch_size=128, filters=1):
    self.dDepth = dDepth
    self.gDepth = gDepth
    self.batch_size = batch_size
    self.filters = filters
    self.image_in = layers.Input(shape=(28,28,1))
    self.noise_in = layers.Input(shape=(100,))
    self.label_in = layers.Input(shape=(10,))
    self.D = self.discriminator(self.image_in,self.label_in)
    self.G = self.generator(self.noise_in,self.label_in)
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    #self.model = tf.keras.Sequential([self.D,self.G])
  
  def discriminator_loss(self,real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss

  def generator_loss(self,fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

  def discriminator(self, image_in, label_in):
    # Prepare images
    x = layers.Conv2D(64*self.filters,(5,5),strides=(1,1),padding='same')(image_in)
    x = Model(inputs=image_in, outputs=x)
    # Prepare labels
    y = layers.Dense(28*28*64)(label_in)
    y = layers.Reshape((28,28,64))(y)
    y = layers.Conv2D(64*self.filters,(5,5),strides=(1,1),padding='same')(y)
    y = Model(inputs=label_in, outputs=y)
    # Combining
    combined = layers.concatenate([x.output, y.output])
    # 28x28x128
    z = layers.Conv2D(256*self.filters,(5,5),strides=(2,2),padding='same')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Conv2D(512*self.filters,(5,5),strides=(2,2),padding='same')(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Flatten()(z)
    z = layers.Dense(4*self.filters)(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)
    z = layers.Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[x.input,y.input], outputs=z)

    #print(model.summary())
    return model

  def generator(self, noise_in, label_in):
    # Noise vector preconditioning
    x = layers.Dense(7*7*256)(noise_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7,7,256))(x)
    x = layers.Conv2DTranspose(256*self.filters,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = Model(inputs=noise_in, outputs=x)
    # Label preconditioning
    y = layers.Dense(7*7*256)(label_in)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.Reshape((7,7,256))(y)
    y = layers.Conv2DTranspose(256*self.filters,(3,3),strides=(1,1),padding='same',use_bias=False)(y)
    y = Model(inputs=label_in, outputs=y)
    
    # Comining both in one input 
    combined = layers.concatenate([x.output,y.output])
    # z = layers.Flatten()(combined)
    # z = layers.Dense(7*7*512)(z)
    # z = layers.BatchNormalization()(z)
    # z = layers.LeakyReLU(alpha=0.2)(z)
    # z = layers.Reshape((7,7,512))(z)

    z = layers.Conv2DTranspose(256*self.filters,(5,5),strides=(1,1),padding='same',use_bias=False)(combined)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)

    z = layers.Conv2DTranspose(128*self.filters,(5,5),strides=(2,2),padding='same',use_bias=False)(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)

    z = layers.Conv2DTranspose(64*self.filters,(5,5),strides=(2,2),padding='same',use_bias=False)(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpha=0.2)(z)

    z = layers.Conv2DTranspose(1,(5,5),strides=(1,1),padding='same',use_bias=False,activation='tanh')(z)

    model = Model(inputs=[x.input,y.input], outputs=z)

    #print(model.summary())
    return model

  @tf.function
  def train_step(self,images,labels,noise_dim):
    noise = tf.random.normal([self.batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.G([noise,labels],training=True)

      real_output = self.D([images,labels],training=True)
      fake_output = self.D([generated_images,labels],training=True)

      gen_loss = self.generator_loss(fake_output)
      total_loss, disc_loss, gen_loss_inv = self.discriminator_loss(real_output,fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(total_loss, self.D.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    return gen_loss, disc_loss

  def train(self,x_train,labels=None,epochs=10,noise_dim=100):
    mean_gen_loss = []
    mean_disc_loss = []

    for epoch in range(epochs):
      start = time.time()
      batch_gen_loss = []
      batch_disc_loss = []

      for image_batch, label_batch in zip(x_train,labels):
        gen_loss, disc_loss = self.train_step(image_batch,label_batch,noise_dim)
        batch_gen_loss.append(gen_loss.numpy())
        batch_disc_loss.append(disc_loss.numpy())

      mean_gen_loss.append(np.mean(batch_gen_loss))
      mean_disc_loss.append(np.mean(batch_disc_loss))

      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      self.generate_and_save_images(self.G,epoch+1,noise_dim)

      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
      print ('Generator loss: {}'.format(mean_gen_loss[epoch]))
      print ('Discriminator loss: {}'.format(mean_disc_loss[epoch]))

    # Generate after the final epoch
    display.clear_output(wait=True)
    self.generate_and_save_images(self.G,epochs,noise_dim)

    return mean_gen_loss, mean_disc_loss

  def generate_and_save_images(self,model,epoch,noise_dim):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    test_input = tf.random.normal([10, noise_dim])
    test_labels = to_categorical(np.arange(10), num_classes=10)

    predictions = model([test_input,test_labels], training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

