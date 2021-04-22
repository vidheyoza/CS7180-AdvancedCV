from keras.utils import plot_model

from models import unet_3layer, simple_gan

unet_model = unet_3layer()
generator, discriminator = simple_gan()

plot_model(unet_model, to_file='saved_models\\unet_3layer.png', dpi=384)
plot_model(generator, to_file='saved_models\\simple_gan_gen.png', dpi=384)
plot_model(discriminator, to_file='saved_models\\simple_gan_disc.png', dpi=384)
