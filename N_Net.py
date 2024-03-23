from keras.models import Model
from keras.layers import Conv2D, Input, Add, Softmax
from keras.layers import MaxPooling2D, AveragePooling2D, Activation
from keras.layers import BatchNormalization, Dropout, Dense
from keras.layers import Flatten
from att import se_block
from custom_function import *


def einet(input_size=(256, 256, 1), drop_rate=0.0, l2=2e-4, bias_init_v=0.2):
    input_img = Input(input_size)
    l2_norm = keras.regularizers.l2(l2)
    bias_init = keras.initializers.Constant(bias_init_v)

    # preprocessing module
    conv_00 = Conv2D(32, 3, padding='same', name='conv_00', kernel_regularizer=l2_norm, bias_initializer=bias_init)(input_img)
    conv_01 = Conv2D(32, 3, padding='same', name='conv_01', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_00)
    bn_01 = BatchNormalization(name='bn_01')(conv_01)

    # Enception #1
    conv_10 = Conv2D(32, (3, 3), padding='same', name='conv_10', kernel_regularizer=l2_norm, bias_initializer=bias_init)(bn_01)
    conv_11 = Conv2D(32, (3, 3), padding='same', name='conv_11', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_10)
    bn_11 = BatchNormalization(name='bn_11')(conv_11)
    relu_11 = Activation('relu')(bn_11)

    concate_res = keras.layers.concatenate([bn_01, relu_11], axis=-1)

    conv_12 = Conv2D(32, (3, 3), padding='same', name='conv_12', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    conv_13 = Conv2D(32, (3, 3), padding='same', name='conv_13', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_12)
    bn_13 = BatchNormalization(name='bn_13')(conv_13)
    relu_13 = Activation('relu')(bn_13)

    concate_res = keras.layers.concatenate([relu_11, relu_13], axis=-1)

    conv_14 = Conv2D(32, (3, 3), padding='same', name='conv_14', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    se_01 = se_block(32, name='se_01')(conv_14)
    conv_15 = Conv2D(32, (3, 3), padding='same', name='conv_15', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_01)
    bn_15 = BatchNormalization(name='bn_15')(conv_15)
    relu_15 = Activation('relu')(bn_15)

    concate_res = keras.layers.concatenate([bn_01, relu_15], axis=-1)

    # Enception #2
    conv_20 = Conv2D(32, (3, 3), padding='same', name='conv_20', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    se_02 = se_block(32, name='se_02')(conv_20)
    conv_21 = Conv2D(32, (3, 3), padding='same', name='conv_21', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_02)
    bn_21 = BatchNormalization(name='bn_21')(conv_21)
    stt_21 = STT(init_theta=1, trainable=True, name='stt_21')(bn_21)

    concate_res = keras.layers.concatenate([concate_res, stt_21], axis=-1)

    conv_22 = Conv2D(32, (3, 3), padding='same', name='conv_22', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    se_03 = se_block(32, name='se_03')(conv_22)
    conv_23 = Conv2D(32, (3, 3), padding='same', name='conv_23', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_03)
    bn_23 = BatchNormalization(name='bn_23')(conv_23)
    stt_23 = STT(init_theta=1, trainable=True, name='stt_23')(bn_23)

    concate_res = keras.layers.concatenate([stt_21, stt_23], axis=-1)

    conv_24 = Conv2D(128, (1, 1), padding='same', name='conv_24', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    conv_25 = Conv2D(128, (3, 3), padding='same', name='conv_25', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_24)
    se_04 = se_block(128, name='se_04')(conv_25)
    conv_26 = Conv2D(64, (1, 1), padding='same', name='conv_26', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_04)
    bn_26 = BatchNormalization(name='bn_26')(conv_26)
    relu_26 = Activation('relu')(bn_26)

    concate_res = keras.layers.concatenate([relu_15, relu_26], axis=-1)

    # Enception #3
    conv_30 = Conv2D(64, (3, 3), padding='same', name='conv_30', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    se_05 = se_block(64, name='se_05')(conv_30)
    conv_31 = Conv2D(64, (3, 3), padding='same', name='conv_31', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_05)
    bn_31 = BatchNormalization(name='bn_31')(conv_31)
    relu_31 = Activation('relu')(bn_31)

    concate_res = keras.layers.concatenate([concate_res, relu_31], axis=-1)

    conv_32 = Conv2D(64, (3, 3), padding='same', name='conv_32', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    se_06 = se_block(64, name='se_06')(conv_32)
    conv_33 = Conv2D(64, (3, 3), padding='same', name='conv_33', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_06)
    bn_33 = BatchNormalization(name='bn_33')(conv_33)
    relu_33 = Activation('relu')(bn_33)

    concate_res = keras.layers.concatenate([relu_31, relu_33], axis=-1)

    conv_34 = Conv2D(128, (1, 1), padding='same', name='conv_34', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    conv_35 = Conv2D(128, (3, 3), padding='same', name='conv_35', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_34)
    se_07 = se_block(128, name='se_07')(conv_35)
    conv_36 = Conv2D(64, (1, 1), padding='same', name='conv_36', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_07)
    bn_36 = BatchNormalization(name='bn_36')(conv_36)
    res = Activation('relu')(bn_36)

    # B-Block #1
    conv_40 = Conv2D(64, (3, 3), padding='same', name='conv_40', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    conv_41 = Conv2D(64, (3, 3), padding='same', name='conv_41', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_40)
    bn_41 = BatchNormalization(name='bn_41')(conv_41)
    relu_41 = Activation('relu')(bn_41)

    res = Add()([res, relu_41])

    conv_42 = Conv2D(128, (1, 1), padding='same', name='conv_42', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_42 = BatchNormalization(name='bn_42')(conv_42)
    pool_42 = MaxPooling2D(pool_size=(5, 5), strides=2, padding='same')(bn_42)

    conv_43 = Conv2D(64, (3, 3), padding='same', name='conv_43', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_43 = BatchNormalization(name='bn_43')(conv_43)
    relu_43 = Activation('relu')(bn_43)
    res = Add()([res, relu_43])
    conv_44 = Conv2D(128, (3, 3), padding='same', name='conv_44', strides=2, kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_44 = BatchNormalization(name='bn_44')(conv_44)

    res = Add()([pool_42, bn_44])

    # B-Block #2
    conv_50 = Conv2D(64, (3, 3), padding='same', name='conv_50', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    conv_51 = Conv2D(64, (3, 3), padding='same', name='conv_51', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_50)
    bn_51 = BatchNormalization(name='bn_51')(conv_51)
    relu_51 = Activation('relu')(bn_51)

    res = Add()([res, relu_51])

    conv_52 = Conv2D(128, (1, 1), padding='same', name='conv_52', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_52 = BatchNormalization(name='bn_52')(conv_52)
    pool_52 = MaxPooling2D(pool_size=(5, 5), strides=2, padding='same')(bn_52)

    conv_53 = Conv2D(64, (3, 3), padding='same', name='conv_53', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_53 = BatchNormalization(name='bn_53')(conv_53)
    relu_53 = Activation('relu')(bn_53)
    res = Add()([res, relu_53])
    conv_54 = Conv2D(128, (3, 3), padding='same', name='conv_54', strides=2, kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_54 = BatchNormalization(name='bn_54')(conv_54)

    res = Add()([pool_52, bn_54])

    # Enception #4
    conv_90 = Conv2D(256, (1, 1), padding='same', name='conv_90', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    conv_91 = Conv2D(256, (3, 3), padding='same', name='conv_91', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_90)
    se_08 = se_block(128, name='se_08')(conv_91)
    conv_92 = Conv2D(128, (1, 1), padding='same', name='conv_92', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_08)
    bn_92 = BatchNormalization(name='bn_92')(conv_92)
    relu_92 = Activation('relu')(bn_92)

    concate_res = keras.layers.concatenate([res, relu_92], axis=-1)

    conv_93 = Conv2D(128, (1, 1), padding='same', name='conv_93', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    conv_94 = Conv2D(128, (3, 3), padding='same', name='conv_94', kernel_regularizer=l2_norm, bias_initializer=bias_init)(conv_93)
    se_09 = se_block(128, name='se_09')(conv_94)
    conv_95 = Conv2D(64, (1, 1), padding='same', name='conv_95', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_09)
    bn_95 = BatchNormalization(name='bn_95')(conv_95)
    relu_95 = Activation('relu')(bn_95)

    concate_res = keras.layers.concatenate([relu_92, relu_95], axis=-1)

    conv_96 = Conv2D(64, (3, 3), padding='same', name='conv_96', kernel_regularizer=l2_norm, bias_initializer=bias_init)(concate_res)
    bn_96 = BatchNormalization(name='bn_96')(conv_96)
    relu_96 = Activation('relu')(bn_96)

    se_10 = se_block(64, name='se_10')(relu_96)

    conv_97 = Conv2D(64, (3, 3), padding='same', name='conv_97', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_10)
    bn_97 = BatchNormalization(name='bn_97')(conv_97)
    relu_97 = Activation('relu')(bn_97)

    res = Add()([res, relu_97])

    # B-Block #3
    conv_60 = Conv2D(64, (3, 3), padding='same', name='conv_60', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_60 = BatchNormalization(name='bn_60')(conv_60)
    relu_60 = Activation('relu')(bn_60)

    se_11 = se_block(64, name='se_11')(relu_60)

    conv_61 = Conv2D(64, (3, 3), padding='same', name='conv_61', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_11)
    bn_61 = BatchNormalization(name='bn_61')(conv_61)
    relu_61 = Activation('relu')(bn_61)

    res = Add()([res, relu_61])

    conv_62 = Conv2D(128, (1, 1), padding='same', name='conv_62', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_62 = BatchNormalization(name='bn_62')(conv_62)
    pool_62 = MaxPooling2D(pool_size=(5, 5), strides=2, padding='same')(bn_62)

    conv_63 = Conv2D(64, (3, 3), padding='same', name='conv_63', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_63 = BatchNormalization(name='bn_63')(conv_63)
    relu_63 = Activation('relu')(bn_63)
    relu_63 = Add()([res, relu_63])
    conv_64 = Conv2D(128, (3, 3), padding='same', name='conv_64', strides=2, kernel_regularizer=l2_norm, bias_initializer=bias_init)(relu_63)
    bn_64 = BatchNormalization(name='bn_64')(conv_64)

    res = Add()([pool_62, bn_64])

    # B-Block #4
    conv_70 = Conv2D(128, (3, 3), padding='same', name='conv_70', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_70 = BatchNormalization(name='bn_70')(conv_70)
    relu_70 = Activation('relu')(bn_70)

    se_12 = se_block(128, name='se_12')(relu_70)

    conv_71 = Conv2D(128, (3, 3), padding='same', name='conv_71', kernel_regularizer=l2_norm, bias_initializer=bias_init)(se_12)
    bn_71 = BatchNormalization(name='bn_71')(conv_71)
    relu_71 = Activation('relu')(bn_71)

    res = Add()([res, relu_71])

    conv_72 = Conv2D(256, (1, 1), padding='same', name='conv_72', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_72 = BatchNormalization(name='bn_72')(conv_72)
    pool_72 = MaxPooling2D(pool_size=(5, 5), strides=2, padding='same')(bn_72)

    conv_73 = Conv2D(128, (3, 3), padding='same', name='conv_73', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_73 = BatchNormalization(name='bn_73')(conv_73)
    relu_73 = Activation('relu')(bn_73)
    relu_73 = Add()([res, relu_73])
    conv_74 = Conv2D(256, (3, 3), padding='same', name='conv_74', strides=2, kernel_regularizer=l2_norm, bias_initializer=bias_init)(relu_73)
    bn_74 = BatchNormalization(name='bn_74')(conv_74)

    res = Add()([pool_72, bn_74])

    # feature fusion
    conv_80 = Conv2D(512, (3, 3), padding='same', name='conv_80', kernel_regularizer=l2_norm, bias_initializer=bias_init)(res)
    bn_80 = BatchNormalization(name='bn_81')(conv_80)
    relu_80 = Activation('relu')(bn_80)

    conv_81 = Conv2D(128, (3, 3), padding='same', name='conv_81', kernel_regularizer=l2_norm, bias_initializer=bias_init)(relu_80)
    bn_81 = BatchNormalization(name='bn_81')(conv_81)
    relu_81 = Activation('relu')(bn_81)
    # LAP
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=4)(relu_81)

    fea_lc = Flatten()(avg_pool)

    if drop_rate:
        fea_lc = Dropout(drop_rate)(fea_lc)

    # classification
    fc_1 = Dense(128, name='fc_1', activation='relu', kernel_regularizer=l2_norm)(fea_lc)
    fc_2 = Dense(2, name='fc_2', kernel_regularizer=l2_norm)(fc_1)
    ip = Softmax()(fc_2)

    model = Model(inputs=input_img, outputs=ip)

    return model