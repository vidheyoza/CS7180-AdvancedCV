# Main training file

import glob
import rawpy
from tqdm import tqdm

from models import unet_3layer
from data import load_raw_images

# root paths and variables

data_root = 'D:\\Terabyte2.0\\Datasets\\See in the Dark\\'
project_root = 'C:\\Users\\vidhe\\Documents\\Northeastern\\4\\AdvancedCV\\SeeInTheDark\\see-in-the-dark\\'

NUM_EPOCHS = 100
BATCH_SIZE = 4

# TODO: load and preprocess dataset

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

# TODO: load and prepare model

# get input shape from one of the images
DATA_SIZE = len(dark_images_paths)

img = rawpy.imread(dark_images_paths[0])
# print(img.sizes)
INPUT_SHAPE = (int(img.sizes.height / 2), int(img.sizes.width / 2), 4)

# load model
unet_model = unet_3layer(INPUT_SHAPE)
unet_model.summary()

# TODO: train model

# batch-wise training means using loops for epochs and batches both

for e in range(NUM_EPOCHS):
    for b in tqdm(range(int(DATA_SIZE / BATCH_SIZE))):
        # load batch data
        X_train, y_train = load_raw_images(BATCH_SIZE, b, dark_images_paths, light_images_dict)

        # train on batch
        unet_model.train_on_batch(X_train, y_train)

# TODO: save model weights
unet_model.save_weights(project_root + 'saved_weights\\unet_3layer_e' +
                        str(NUM_EPOCHS) + 'b' + str(BATCH_SIZE) + '.h5')
