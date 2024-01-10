import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


print("Num GPUs avail: ", len(tf.config.list_physical_devices('GPU')))



ds = tfds.load('fashion_mnist', split='train')



def scale_images(data):
     image = data['image']
     return image/255

ds = tfds.load('fashion_mnist', split='train')

#preprocessing data pipeline
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)

#creates the generator model
gen = Sequential()

#block 1
gen.add(Dense(7*7*128, input_dim=128))
gen.add(LeakyReLU(0.2))
gen.add(Reshape((7,7,128)))

#block 2
gen.add(UpSampling2D())
gen.add(Conv2D(128,5,padding='same'))
gen.add(LeakyReLU(0.2))

#block 3
gen.add(UpSampling2D())
gen.add(Conv2D(128,5,padding='same'))
gen.add(LeakyReLU(0.2))

#block 4
gen.add(Conv2D(128,4,padding='same'))
gen.add(LeakyReLU(0.2))

#block 5
gen.add(Conv2D(128,4,padding='same'))
gen.add(LeakyReLU(0.2))

#conv layer
gen.add(Conv2D(1,4,padding='same', activation='sigmoid'))
##############################



##fig, ax = plt.subplots(ncols=4,figsize=(20,20))
##
##for idx, img in enumerate(img):
##    ax[idx].imshow(np.squeeze(img))
##    ax[idx].title.set_text(idx)
##
##plt.show()



#Creates discriminator model
dis = Sequential()


#block 1
dis.add(Conv2D(32,5,input_shape=(28,28,1)))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.4))

#block 2
dis.add(Conv2D(64,5))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.4))

#block 3
dis.add(Conv2D(128,5))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.4))

#block 4
dis.add(Conv2D(256,5))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.4))

#block 5
dis.add(Flatten())
dis.add(Dropout(0.4))
dis.add(Dense(1,activation='sigmoid'))

############################



g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


class FashionGAN(Model):
    def __init__(self, gen, dis, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.gen = gen
        self.dis = dis
    
    def train_step(self, batch):
        real_imgs = batch
        fake_imgs = self.gen(tf.random.normal((128,128,1)), training=False)

        with tf.GradientTape() as d_tape:
            yhat_real = self.dis(real_imgs, training=True)
            yhat_fake = self.dis(fake_imgs, training=True)
            yhat_realfake = tf.concat([yhat_real,yhat_fake], axis=0)

            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)


            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real,noise_fake],axis=0)


            total_d_loss = self.d_loss(y_realfake,yhat_realfake)

        dgrad = d_tape.gradient(total_d_loss, self.dis.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad,self.dis.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.gen(tf.random.normal((128,128,1)), training=True)

            predicted_labels = self.dis(gen_images, training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad,self.gen.trainable_variables))

        return {"d_loss":total_d_loss, "g_loss":total_g_loss}
    
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args,**kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss



#initializes the gan model
fashgan = FashionGAN(gen, dis)
fashgan.compile(g_opt,d_opt,g_loss,d_loss)


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self,epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        gen_imgs = self.model.gen(random_latent_vectors)
        gen_imgs *= 255
        gen_imgs.numpy()
        for i in range(self.num_img):
            img = array_to_img(gen_imgs[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))


#2000 is recommended
hist = fashgan.fit(ds,epochs=20,callbacks=[ModelMonitor()])














