from keras.layers import (Activation, Dense, GlobalAveragePooling2D, Reshape, multiply)


def se_block(input_feature, ratio=16, name=""):
    channel = input_feature._keras_shape[-1]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)

    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=False,
                       name="se_block_one_" + str(name))(se_feature)

    se_feature = Dense(channel,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       name="se_block_two_" + str(name))(se_feature)
    se_feature = Activation('sigmoid')(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature