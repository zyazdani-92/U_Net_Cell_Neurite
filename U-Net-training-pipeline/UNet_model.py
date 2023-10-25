'''Unet model for keras, written by:
Paul-Louis Pröve
https://github.com/pietz/unet-keras

Modified by: Zahra Yazdani

'''

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization


def conv_block(inp, ch, act='relu', dropout=0.5, batchnorm=False):
    x = Conv2D(ch, 3, activation=act, padding='same')(inp)
    if batchnorm:
        x = BatchNormalization()(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = Conv2D(ch, 3, activation=act, padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    return x


def unet(img_shape, out_ch=1, start_ch=64, layer=4, inc_rate=2.0, activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    inp = Input(shape=img_shape)
    x = inp
    
    # Encoding path
    conv_layers = []
    for i in range(layer):
        ch = int(start_ch * inc_rate**i)
        x = conv_block(x, ch, activation, dropout, batchnorm)
        conv_layers.append(x)
        if maxpool and i != layer-1:
            x = MaxPooling2D()(x)
    
    # Decoding path
    for i in reversed(range(layer-1)):
        ch = int(start_ch * inc_rate**(i+1))
        x = UpSampling2D()(x)
        x = Concatenate()([conv_layers[i], x])
        x = conv_block(x, ch, activation, dropout, batchnorm)
    
    # Output
    out = Conv2D(out_ch, 1, activation='sigmoid')(x)
    
    # Model
    model = Model(inputs=inp, outputs=out)
    
    return model

