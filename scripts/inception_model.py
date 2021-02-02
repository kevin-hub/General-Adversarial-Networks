import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time

class LeNet():
  def __init__(self, lr = 0.001, name = "LeNet"):
    # Build and compile the basic lenet model
    self.model = self.build_model()
    self.optimizer = tf.keras.optimizers.Adam(1e-4)

  def build_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    return model

  def discriminator_loss(self,real_output):
    return cross_entropy(tf.ones_like(real_output), real_output)

  @tf.function
  def train_step(self,images,labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      real_output = self.model(images, training=True)
      disc_loss = self.discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.trainable_variables)

    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.trainable_variables))

  def train(self,x_train,y_train,epochs=25):
    for epoch in range(epochs):
      start = time.time()

      for image_batch, label_batch in zip(x_train,y_train):
        self.train_step(image_batch,label_batch)

      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
