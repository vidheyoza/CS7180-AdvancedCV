# Learning to See in the Dark

This is the Learning to See in the Dark project code directory. All code and saved weights are in `model_weights`. 

## Abstract

Capturing low-light images is difficult particularly in daily-use devices like smartphones, where long-exposure or multi-exposure HDR captures are prone to shaking and bad post-processing. 
Using AI-based techniques requires a light model footprint, and hence developing small-scale generative models show strong potential in this problem domain. 
We propose a simple GAN-based training architecture that makes use of the prowess of supervised learning using the See-in-the-Dark Dataset, which contains dark-bright image pairs across different image contexts, to generate light dynamically on a dark image input to the generator. 
Results show PSNR of 29.07 and SSIM of 0.791, which fall in line with other benchmark techniques in the problem domain. 

## How to understand the code

- `data.py` and `models.py` contain helper functions for loading the data and model respectively. 
- `training_XXX.py` files are used to train the respective models. 
  - `unet_3layer` is a simple U-Net with a 3-layer encoder-decoder architecture.
  - `simple_gan` is a GAN-based model set that uses a U-Net like generator and a CNN-classifier based discriminator. 
- `testing.py` is for testing the pre-trained models. 

## How to run the code

THe SITD dataset is too large to be pushed on GitHub, so to get the data visit these links: [Fuji](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip) and [Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip). 
Alternatively, use the `download_dataset.py` script in [SeeInTheDark/Learning-to-See-in-the-Dark]. 
- For training, modify the `data_root` path string in `training_gan.py` to where the data is downloaded and extracted. 
- For testing, modify the `data_root` path string in `testing_gan.py` similarly. 
