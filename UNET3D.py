from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.callbacks import CSVLogger
import math

import numpy as np

img_rows = 168
img_cols = 168
img_depth = 168

smooth = 1.

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_net_multiPlane():
    filterFactor = 1
    #### tra branch #####
    inputs_tra = Input((168, 168, 168, 1))
    conv1_tra = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    conv1_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    conv2_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    conv3_tra = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((168, 168, 168, 1))
    conv1_cor = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    conv1_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    conv2_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    conv3_cor = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####
    
    inputs_sag = Input((168, 168, 168, 1))
    conv1_sag = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    conv1_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    conv2_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    conv3_sag = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv4)

    up6 = Conv3DTranspose(128,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv4)
    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])
    conv6 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3DTranspose(64,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv6)
    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])
    conv7 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3DTranspose(32,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv7)
    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])
    conv8 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model


def get_net_singlePlane():

    filterFactor = 1

    inputs = Input((img_depth, img_rows, img_cols, 1))
    conv1 = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv4)

    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv4)
    up6 = concatenate([up6, conv3])
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv6)
    up7 = concatenate([up7, conv2])
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)
    up8 = concatenate([up8, conv1])
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


