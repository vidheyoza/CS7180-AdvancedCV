# Main training file for GAN

import numpy as np
import glob
import rawpy
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from models import simple_gan
from data import load_raw_images

# TF GPU flags
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# root paths and variables

data_root = 'D:\\Terabyte2.0\\Datasets\\See in the Dark\\'
project_root = 'C:\\Users\\vidhe\\Documents\\Northeastern\\4\\AdvancedCV\\SeeInTheDark\\see-in-the-dark\\'

NUM_EPOCHS = 1
BATCH_SIZE = 1

# #### Load and preprocess dataset

# data paths
input_dir = data_root + 'Sony\\Sony\\short\\'
gt_dir = data_root + 'Sony\\Sony\\long\\'

dark_images_paths = glob.glob(input_dir + '*.ARW')
light_images_paths = glob.glob(gt_dir + '*.ARW')

# index light images for data augmentation
light_images_dict = {}
for path in light_images_paths:
    img_name = path.split('\\')[-1].split('_')[0]
    light_images_dict[img_name] = path

# since data volume is large, batch-wise loading is done

# #### Load and prepare model

# 1. get input shape from one of the images
DATA_SIZE = len(dark_images_paths)

img = rawpy.imread(dark_images_paths[0])
# INPUT_SHAPE = (int(img.sizes.height / 2), int(img.sizes.width / 2), 4)

INPUT_SHAPE = (100, 100, 4)
print(INPUT_SHAPE)

# 2. load optimizer and loss
opt = Adam(lr=0.0002, beta_1=0.5)
loss = 'binary_crossentropy'

generator, discriminator = simple_gan(INPUT_SHAPE, gen_loss=loss, gen_opt=opt, disc_loss=loss, disc_opt=opt)
generator.summary()
discriminator.summary()

# 3. build GAN
discriminator.trainable = False
gan_input = Input(shape=INPUT_SHAPE)
gan = generator(gan_input)
gan_output = discriminator(gan)
gan_model = Model(gan_input, gan_output)
gan_model.compile(loss=loss, optimizer=opt)

# #### Train model
# '''
# batch-wise training means using loops for epochs and batches both
# 2-step training: 1 = min generator loss, 2 = max discriminator loss
gen_losses = []
disc_losses = []
for e in range(NUM_EPOCHS):
    for b in tqdm(range(int(DATA_SIZE / BATCH_SIZE))):
        # 1. get real images from storage, generated image from generator
        real_dark_images, real_light_images = load_raw_images(BATCH_SIZE, b, dark_images_paths, light_images_dict,
                                                              patched_images=True, patch_size=100, patch_stride=100)

        # noise_shape = real_light_images.shape
        # noise = np.random.normal(0, 1, size=list(noise_shape))
        generated_images = generator.predict(real_dark_images)

        # verifying images are of same shape
        # print(generated_images.shape)
        # print(real_dark_images.shape)

        # 2. train discriminator on fused dataset
        discriminator_X = np.concatenate((real_light_images, generated_images), axis=0)
        discriminator_y = np.zeros(2 * generated_images.shape[0])
        discriminator_y[:BATCH_SIZE] = 0.9

        discriminator.trainable = True
        disc_loss = discriminator.train_on_batch(discriminator_X, discriminator_y)

        # 3. train generator with goal of maximizing discriminator loss
        discriminator.trainable = False
        gan_y = np.ones(generated_images.shape[0])
        gen_loss = gan_model.train_on_batch(real_dark_images, gan_y)

    gen_losses.append(gen_loss)
    disc_losses.append(disc_loss)

    # TODO plot sample images to see progress after every X epochs

# #### Save model weights
gan_model.save_weights(project_root + 'saved_weights\\simplegan_patched_e' +
                       str(e) + 'b' + str(BATCH_SIZE) + '.h5')
# '''

# #### Test model

# load saved model
generator, discriminator = simple_gan(INPUT_SHAPE, gen_loss=loss, gen_opt=opt, disc_loss=loss, disc_opt=opt)
discriminator.trainable = False
gan_input = Input(shape=INPUT_SHAPE)
gan = generator(gan_input)
gan_output = discriminator(gan)
gan_model = Model(gan_input, gan_output)
gan_model.load_weights(project_root + 'saved_weights\\simplegan_patched_e1b1.h5')
gan_model.compile(loss=loss, optimizer=opt)

# test images
real_dark_images, real_light_images = load_raw_images(10, 500, dark_images_paths, light_images_dict,
                                                              patched_images=True, patch_size=100, patch_stride=100)
predicted = generator.predict(real_dark_images)
