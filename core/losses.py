import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np

from .networks import img_size


# img_size = 512

image_shape = (img_size, img_size, 3)


def l2_loss(y_true, y_pred):
    return K.mean((y_pred - y_true)**2)


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def perceptual_and_l2_loss(y_true, y_pred):
    return 0.5*perceptual_loss(y_true, y_pred) + 0.5*l2_loss(y_true, y_pred)
