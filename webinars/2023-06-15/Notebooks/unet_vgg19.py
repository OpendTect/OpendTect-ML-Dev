import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Input, Add, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from keras.optimizers import *
from keras import backend as K


def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true,
        pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def UNet_VGG19():
    """ Pre-trained VGG19 unetvgg_model """
    unetvgg_model = VGG19(include_top=False, weights="imagenet")

    vgg19_weights = {}
    for layer in unetvgg_model.layers[2:]:
        if "conv" in layer.name:
            vgg19_weights["1024_" + layer.name] = unetvgg_model.get_layer(layer.name).get_weights()

    del unetvgg_model

    inputs = Input(shape=(128, 128, 1), name='1024_input')

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1), data_format="channels_last", name='1024_block1_conv1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='1024_block1_conv2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block1_pool')(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='1024_block2_conv1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='1024_block2_conv2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block2_pool')(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv2')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block3_pool')(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv2')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv3')(conv4)

    up5 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(conv5)
    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(conv6)
    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(conv7)
    conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    unetvgg_model = Model(inputs=[inputs], outputs=[conv8])

    for layer in unetvgg_model.layers[2:]:
        if ('1024' in layer.name and 'conv' in layer.name):
            unetvgg_model.get_layer(layer.name).set_weights(vgg19_weights[layer.name])


    unetvgg_model.summary()
    unetvgg_model.compile(optimizer = Adam(learning_rate = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])
    return unetvgg_model
