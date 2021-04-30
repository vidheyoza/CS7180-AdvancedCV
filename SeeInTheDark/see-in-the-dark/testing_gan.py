# Testing the trained GAN model

import numpy as np
import glob
import rawpy
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from models import simple_gan
from data import load_raw_images, bayer_to_jpeg

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

    DATA_SIZE = len(dark_images_paths)

    img = rawpy.imread(dark_images_paths[0])
    # INPUT_SHAPE = (int(img.sizes.height / 2), int(img.sizes.width / 2), 4)

INPUT_SHAPE = (100, 100, 4)
print(INPUT_SHAPE)

# #### Test model
opt = Adam(lr=0.0002, beta_1=0.5)
loss = 'binary_crossentropy'

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

# loading random batches for testing evenly across data
real_dark_images, real_light_images = load_raw_images(10, 500, dark_images_paths, light_images_dict,
                                                      patched_images=True, patch_size=100, patch_stride=100)
predicted = generator.predict(real_dark_images)

# visualize predicted images

for i, image in enumerate(predicted):
    plt.imsave('sample_outputs/' + str(i) + '.jpg', bayer_to_jpeg(image))
