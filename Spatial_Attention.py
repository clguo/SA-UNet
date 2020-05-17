from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D
from Dropblock import  *
def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def eca_block(input_feature, kernel=7):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    eca_feature = GlobalAveragePooling2D()(input_feature)
    eca_feature = Reshape((1, 1, channel))(eca_feature)
    assert eca_feature._keras_shape[1:] == (1, 1, channel)
    eca_feature = Permute((3, 1, 2))(eca_feature)
    eca_feature = Lambda(squeeze)(eca_feature)


    eca_feature = Conv1D(filters=1,kernel_size=kernel,strides=1,padding="same")(eca_feature)
    eca_feature=Lambda(unsqueeze)(eca_feature)


    eca_feature = Permute((2, 3, 1))(eca_feature)
    eca_feature = Activation('relu')(eca_feature)
    assert eca_feature._keras_shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        eca_feature = Permute((3, 1, 2))(eca_feature)

    eca_feature = multiply([input_feature, eca_feature])
    return eca_feature

def unsqueeze(input):
    return K.expand_dims(input,axis=-1)

def squeeze(input):
    return K.squeeze(input,axis=-1)

