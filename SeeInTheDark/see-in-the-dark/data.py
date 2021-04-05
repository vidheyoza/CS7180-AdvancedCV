# Data loading pipelines For RAW and jpg files both
# DISCLAIMER: models trained on RAW will not work on jpg in inference (or vice-versa)
# because RAW images are 4-channel

from typing import List
import numpy as np
import rawpy
from PIL import Image


def pack_raw(raw):
    """
    Pack RAW image loaded from rawpy

    @param raw: RAW image
    @return: 4-channel Bayer image in NP format
    """

    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    # subtract the black level
    im = np.maximum(im - 512, 0) / (16383 - 512)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    h = img_shape[0]
    w = img_shape[1]

    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)
    return out


def load_raw_from_list(list_of_raw_files: List[str]) -> np.ndarray:
    """
    Load list of RAW images from list of file paths

    @param list_of_raw_files: list of string file paths
    @return: list of images in NP format
    """

    raw_images = []
    for path in list_of_raw_files:
        raw_images.append(pack_raw(rawpy.imread(path)))

    return np.array(raw_images)


def load_raw_images(batch_size, batch_num, dark_images_paths, light_images_dict):
    dark_images = load_raw_from_list(dark_images_paths[batch_num * batch_size:(batch_num + 1) * batch_size])

    light_images_paths = []
    for path in dark_images_paths:
        dark_img_name = path.split('\\')[-1].split('_')[0]
        light_images_paths.append(light_images_dict[dark_img_name])
    light_images = load_raw_from_list(light_images_paths[batch_num * batch_size:(batch_num + 1) * batch_size])

    return dark_images, light_images


def load_img_from_list(list_of_img_files: List[str]) -> np.ndarray:
    raise NotImplementedError('Not yet implemented')
