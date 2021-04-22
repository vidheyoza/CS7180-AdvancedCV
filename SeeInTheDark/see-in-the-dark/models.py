# DISCLAIMER: models here are made on top of TF2.4, so check for dependency
# errors if models are not loading or running as expected.

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, concatenate


def unet_3layer(input_shape=(4240, 2832, 4), loss='mse', optimizer='adam') -> Model:
    """
    Basic U-Net type model with 3-layer encoder-decoder architecture

    @param input_shape: input shape, as dictated by training data
    @param loss: loss function (default MSE)
    @param optimizer: optimization function (default Adam)

    @return: model compiled with given loss and optimizer
    """

    input_layer = Input(input_shape)

    # encoder
    en1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    down1 = MaxPooling2D((2, 2))(en1)

    en2 = Conv2D(32, (3, 3), activation='relu', padding='same')(down1)
    down2 = MaxPooling2D((2, 2))(en2)

    en3 = Conv2D(64, (3, 3), activation='relu', padding='same')(down2)
    down3 = MaxPooling2D((2, 2))(en3)

    # decoder
    de3 = Conv2D(64, (3, 3), activation='relu', padding='same')(down3)
    de3 = UpSampling2D((2, 2))(de3)
    merge3 = concatenate([en3, de3])

    de2 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge3)
    de2 = UpSampling2D((2, 2))(de2)
    merge2 = concatenate([en2, de2])

    de1 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge2)
    de1 = UpSampling2D((2, 2))(de1)
    merge1 = concatenate([en1, de1])

    # final output convolutions
    output_layer = Conv2D(16, (3, 3), activation='relu', padding='same')(merge1)
    output_layer = Conv2D(1, (1, 1), activation='relu', padding='same')(output_layer)

    # model
    model = Model(input_layer, output_layer)
    model.compile(loss=loss, optimizer=optimizer)

    return model


# TODO design a simple GAN architecture
def simple_gan(input_shape=(4240, 2832, 4),
               gen_loss='mse', gen_opt='adam',
               disc_loss='mse', disc_opt='adam') -> (Model, Model):
    """
    Basic GAN with a U-Net based generator and 5-layer CNN discriminator

    @param input_shape: input shape, as dictated by training data
    @param gen_loss: generator loss function (default MSE)
    @param gen_opt: generator optimization function (default Adam)
    @param disc_loss: discriminator loss function (default MSE)
    @param disc_opt: discriminator optimization function (default Adam)

    @return: tuple of models in (G, D) format
    """

    # #### Generator model
    gen_input = Input(input_shape)

    # encoder
    en1 = Conv2D(16, (3, 3), activation='relu', padding='same')(gen_input)
    down1 = MaxPooling2D((2, 2))(en1)

    en2 = Conv2D(32, (3, 3), activation='relu', padding='same')(down1)
    down2 = MaxPooling2D((2, 2))(en2)

    # decoder
    de2 = Conv2D(32, (3, 3), activation='relu', padding='same')(down2)
    de2 = UpSampling2D((2, 2))(de2)
    merge2 = concatenate([en2, de2])

    de1 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge2)
    de1 = UpSampling2D((2, 2))(de1)
    merge1 = concatenate([en1, de1])

    # final output convolutions
    gen_output = Conv2D(16, (3, 3), activation='relu', padding='same')(merge1)
    gen_output = Conv2D(4, (1, 1), activation='relu', padding='same')(gen_output)

    gen = Model(gen_input, gen_output)
    gen.compile(optimizer=gen_opt, loss=gen_loss)

    # #### Discriminator model
    disc_input = Input(input_shape)
    en1 = Conv2D(16, (7, 7), strides=(3, 3), activation='relu')(disc_input)
    down1 = MaxPooling2D((2, 2))(en1)

    en2 = Conv2D(32, (5, 5), strides=(3, 3), activation='relu')(down1)
    down2 = MaxPooling2D((2, 2))(en2)

    fc1 = Flatten()(down2)
    fc1 = Dense(64, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(128, activation='relu')(fc1)
    fc2 = Dropout(0.25)(fc2)

    disc_output = Dense(1, activation='sigmoid')(fc2)

    disc = Model(disc_input, disc_output)
    disc.compile(optimizer=disc_opt, loss=disc_loss)

    return gen, disc


# TODO design a final GAN model to be used
def final_gan() -> (Model, Model):
    raise NotImplementedError('Not yet implemented')
