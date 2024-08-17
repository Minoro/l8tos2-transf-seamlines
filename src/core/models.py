
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, Input, concatenate, 
    AveragePooling2D, Concatenate, UpSampling2D, LayerNormalization, Dense, DepthwiseConv2D, Add, ReLU,
    GlobalAveragePooling2D, Reshape, multiply
)

from tensorflow.keras import Model
import numpy as np

def convolution_block(block_input, n_filters=256, kernel_size=3, dilation_rate=1, use_bias=False, name=None):
    x = Conv2D(
        n_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        name=name
    )(block_input)
    
    bn_name = None
    act_name = None
    if name is not None:
        bn_name = f'bn_{name}'
        act_name = f'act_{name}'

    x = BatchNormalization(name=bn_name)(x)

    return tf.nn.relu(x, name=act_name)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input, n_filters=256):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, n_filters=n_filters, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, n_filters=n_filters, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, n_filters=n_filters, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, n_filters=n_filters, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, n_filters=n_filters, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes, freeze_feature_extractor=False):
    model_input = Input(shape=(image_size, image_size, 3))
    # preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = tf.keras.applications.ResNet50(
        # weights="imagenet", include_top=False, input_tensor=preprocessed
        weights=None, include_top=False, input_tensor=model_input
    )

    if freeze_feature_extractor:
        resnet50.trainable = False

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    # Decoder
    input_a = UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
        name='decoder_upscale_1'
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, n_filters=48, kernel_size=1, name='decoder_conv_low_level_features')

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, name='decoder_conv_2')
    x = convolution_block(x, name='decoder_conv_3')
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    if num_classes == 1:
        model_output = Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid')(x)
    else:
        model_output = Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)

    return Model(inputs=model_input, outputs=model_output, name='deeplabv3')



def single_convolutional_block(inputs, n_filters, kernel_size, strides=1, norm=True, activation=None, name=None):
    x = Conv2D(n_filters, kernel_size, strides, padding='same', name=name)(inputs)
    if name is not None:
        name = f'{name}_bn'
    x = BatchNormalization(momentum=.99, epsilon=1e-5, name=name)(x) if norm is True else x
    x = Activation(activation)(x) if activation else x
    return x


def decoder_mlp(inputs, embed_dim, name=None):
    """
    input shape -> [batches, height, width, embed_dim]
    :return:  shape -> [batches, n_patches, embed_dim]
    """

    if name is not None:
        name = f'{name}_dense'

    batches, height, width, channels = inputs.shape
    x = tf.reshape(inputs, shape=[-1, height * width, channels])
    x = Dense(embed_dim, use_bias=True, name=name)(x)

    return x


def seg_former_decoder_block(inputs, embed_dim, up_size=(4, 4), name=None):
    """
    inputs: shape -> [batches, height, width, embed_dim]
    :return: shape -> [batches, height, width, embed_dim]
    """
    batches, height, width, channels = inputs.shape
    x = decoder_mlp(inputs, embed_dim, name)
    x = tf.reshape(x, shape=[-1, height, width, embed_dim])
    if name is not None:
        name = f'{name}_upsampling'
    x = UpSampling2D(size=up_size, interpolation='bilinear', name=name)(x)

    return x


def seg_former_head(features, embed_dim, num_classes, drop_rate=0.):
    assert len(features) == 4
    assert len(set(feature.shape for feature in features)) == 1

    x = Concatenate(axis=-1)(features)
    x = single_convolutional_block(x, n_filters=embed_dim, kernel_size=1, norm=True, activation='relu', name='segformer_head')
    x = Dropout(rate=drop_rate)(x)

    if num_classes == 1:
        x = Conv2D(num_classes, kernel_size=1, activation='sigmoid', name='segformer_output')(x)
    else:
        x = Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    
    return x




def overlap_patch_embedding(inputs, n_filters, kernel_size, strides):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    print('overlap patch embedding, x.shape: {}'.format(x.shape))
    batches, height, width, embed_dim = x.shape
    x = tf.reshape(x, shape=[-1, height * width, embed_dim])

    return LayerNormalization()(x), height, width


def efficient_multi_head_attention(inputs, height, width, embed_dim, n_heads, scaler=None,
                                   use_bias=True, sr_ratio: int = 1,
                                   attention_drop_rate=0., projection_drop_rate=0.):
    batches, n_patches, channels = inputs.shape
    assert channels == embed_dim
    assert height and width and height * width == n_patches

    head_dim = embed_dim // n_heads
    scaler = head_dim ** -0.5 if scaler is None else scaler

    query = Dense(embed_dim, use_bias=use_bias)(inputs)
    query = tf.reshape(query, shape=[-1, n_patches, n_heads, head_dim])
    query = tf.transpose(query, perm=[0, 2, 1, 3])

    if sr_ratio > 1:
        inputs = tf.reshape(inputs, shape=[-1, height, width, embed_dim])
        # shape -> [batches, height/sr, width/sr, embed_dim]
        inputs = Conv2D(embed_dim, kernel_size=sr_ratio, strides=sr_ratio, padding='same')(inputs)
        inputs = LayerNormalization()(inputs)
        # shape -> [batches, height * width/sr ** 2, embed_dim]
        inputs = tf.reshape(inputs, shape=[-1, (height * width) // (sr_ratio ** 2), embed_dim])

    key_value = Dense(embed_dim * 2, use_bias=use_bias)(inputs)
    if sr_ratio > 1:
        key_value = tf.reshape(key_value, shape=[-1, (height * width) // (sr_ratio ** 2), 2, n_heads, head_dim])
    else:
        key_value = tf.reshape(key_value, shape=[-1, n_patches, 2, n_heads, head_dim])
    key_value = tf.transpose(key_value, perm=[2, 0, 3, 1, 4])
    key, value = key_value[0], key_value[1]

    alpha = tf.matmul(a=query, b=key, transpose_b=True) * scaler
    alpha_prime = tf.nn.softmax(alpha, axis=-1)
    alpha_prime = Dropout(rate=attention_drop_rate)(alpha_prime)

    b = tf.matmul(alpha_prime, value)
    b = tf.transpose(b, perm=[0, 2, 1, 3])
    b = tf.reshape(b, shape=[-1, n_patches, embed_dim])

    x = Dense(embed_dim, use_bias=use_bias)(b)
    x = Dropout(rate=projection_drop_rate)(x)

    return x


def mixed_feedforward_network(inputs, height, width, embed_dim, expansion_rate=4, drop_rate=0., ):
    batches, n_patches, channels = inputs.shape
    assert n_patches == height * width and channels == embed_dim

    x = Dense(int(embed_dim * expansion_rate), use_bias=True)(inputs)
    x = tf.reshape(x, shape=[-1, height, width, int(embed_dim * expansion_rate)])
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = tf.reshape(x, shape=[-1, n_patches, int(embed_dim * expansion_rate)])
    x = Activation('gelu')(x)
    x = Dense(embed_dim, use_bias=True)(x)
    x = Dropout(rate=drop_rate)(x)

    return x


def seg_former_encoder_block(inputs, height, width, embed_dim, n_heads=8, sr_ratio=1, expansion_rate=4,
                             attention_drop_rate=0., projection_drop_rate=0., drop_rate=0.):
    x = LayerNormalization()(inputs)
    x = efficient_multi_head_attention(x, height, width, embed_dim, n_heads=n_heads, sr_ratio=sr_ratio,
                                       attention_drop_rate=attention_drop_rate,
                                       projection_drop_rate=projection_drop_rate)
    branch1 = Add()([inputs, x])
    x = LayerNormalization()(branch1)
    x = mixed_feedforward_network(x, height, width, embed_dim, expansion_rate, drop_rate)
    x = Add()([branch1, x])

    return x



def SegFormer(input_shape,
              num_classes,
              n_blocks=None,
              embed_dims=None,
              decoder_embed_dim=256,
              patch_sizes=None,
              strides=None,
              heads=None,
              reduction_ratios=None,
              expansion_rate=None,
              attention_drop_rate=0.,
              drop_rate=0.,
              ):
    if expansion_rate is None:
        expansion_rate = [8, 8, 4, 4]
    if reduction_ratios is None:
        reduction_ratios = [8, 4, 2, 1]
    if heads is None:
        heads = [1, 2, 4, 8]
    if strides is None:
        strides = [4, 2, 2, 2]
    if patch_sizes is None:
        patch_sizes = [7, 3, 3, 3]
    if embed_dims is None:
        embed_dims = [32, 64, 160, 256]
    if n_blocks is None:
        n_blocks = [2, 2, 2, 2]

    block_range = np.cumsum([0] + n_blocks)
    attention_scheduler = np.linspace(0, attention_drop_rate, num=sum(n_blocks))
    projection_scheduler = np.linspace(0, drop_rate, num=sum(n_blocks))

    inputs = Input(input_shape)

    # encoder
    # stage 1
    x, height1, width1 = overlap_patch_embedding(inputs, embed_dims[0],
                                                 kernel_size=patch_sizes[0], strides=strides[0])

    for index in range(n_blocks[0]):
        attention_range = attention_scheduler[block_range[0]: block_range[1]]
        projection_range = projection_scheduler[block_range[0]: block_range[1]]
        x = seg_former_encoder_block(x, height1, width1, embed_dims[0], heads[0],
                                     sr_ratio=reduction_ratios[0],
                                     expansion_rate=expansion_rate[0],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature1 = tf.reshape(x, shape=[-1, height1, width1, embed_dims[0]])

    # stage 2
    x, height2, width2 = overlap_patch_embedding(feature1, embed_dims[1],
                                                 kernel_size=patch_sizes[1], strides=strides[1])

    for index in range(n_blocks[1]):
        attention_range = attention_scheduler[block_range[1]: block_range[2]]
        projection_range = projection_scheduler[block_range[1]: block_range[2]]
        x = seg_former_encoder_block(x, height2, width2, embed_dims[1], heads[1],
                                     sr_ratio=reduction_ratios[1],
                                     expansion_rate=expansion_rate[1],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature2 = tf.reshape(x, shape=[-1, height2, width2, embed_dims[1]])

    # stage 3
    x, height3, width3 = overlap_patch_embedding(feature2, embed_dims[2],
                                                 kernel_size=patch_sizes[2], strides=strides[2])
    for index in range(n_blocks[2]):
        attention_range = attention_scheduler[block_range[2]: block_range[3]]
        projection_range = projection_scheduler[block_range[2]: block_range[3]]
        x = seg_former_encoder_block(x, height3, width3, embed_dims[2], heads[2],
                                     sr_ratio=reduction_ratios[2],
                                     expansion_rate=expansion_rate[2],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature3 = tf.reshape(x, shape=[-1, height3, width3, embed_dims[2]])

    # stage 4
    x, height4, width4 = overlap_patch_embedding(feature3, embed_dims[3],
                                                 kernel_size=patch_sizes[3], strides=strides[3])
    for index in range(n_blocks[3]):
        attention_range = attention_scheduler[block_range[3]: block_range[4]]
        projection_range = projection_scheduler[block_range[3]: block_range[4]]
        x = seg_former_encoder_block(x, height4, width4, embed_dims[3], heads[3],
                                     sr_ratio=reduction_ratios[3],
                                     expansion_rate=expansion_rate[3],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature4 = tf.reshape(x, shape=[-1, height4, width4, embed_dims[3]])

    feature1 = seg_former_decoder_block(feature1, decoder_embed_dim, up_size=(4, 4), name='segdecoder_1')
    feature2 = seg_former_decoder_block(feature2, decoder_embed_dim, up_size=(8, 8), name='segdecoder_2')
    feature3 = seg_former_decoder_block(feature3, decoder_embed_dim, up_size=(16, 16), name='segdecoder_3')
    feature4 = seg_former_decoder_block(feature4, decoder_embed_dim, up_size=(32, 32), name='segdecoder_4')

    x = seg_former_head([feature1, feature2, feature3, feature4], decoder_embed_dim, num_classes, drop_rate)
    model = Model(inputs, x, name='segformer')

    return model


def SegFormerB0(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[2, 2, 2, 2], embed_dims=[32, 64, 120, 256],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB1(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[2, 2, 2, 2], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB2(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 3, 6, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB3(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 3, 18, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB4(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB5(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 6, 40, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[4, 4, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)





def get_unet(input_height=256, input_width=256, n_filters = 64, dropout = 0.1, batchnorm = True, n_channels=3):
    input_img = Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = tf.keras.Model(inputs=[input_img], outputs=[outputs], name="U-net")
    return model
    

def freeze_unet_layers(unet, mode):

    if mode == 'freeze_all':
        unet.trainable = False
        for layer in unet.layers:
            layer.trainable = False
    
    elif mode == 'freeze_encoder':
        for layer in unet.layers[:39]:
            layer.trainable = False

    elif mode == 'freeze_decoder':
        for layer in unet.layers[39:]:
            layer.trainable = False

    elif mode == 'unfreeze':
        unet.trainable = True
        for layer in unet.layers:
            layer.trainable = True
    
    else:
        raise 'Configuração não conhecida'

    return unet



def freeze_deeplabv3_layers(deeplabv3, mode):
    if mode == 'freeze_all':
        deeplabv3.trainable = False
        for layer in deeplabv3.layers:
            layer.trainable = False
            
    elif mode == 'freeze_encoder':
        for layer in deeplabv3.layers:
            # Descongela o modelo até a primeira camada 
            if layer.name == 'decoder_upscale_1':
                break
            layer.trainable = False

    elif mode == 'unfreeze':
        deeplabv3.trainable = True
        for layer in deeplabv3.layers:
            layer.trainable = True
            
    elif mode == 'freeze_feature_extractor':
        weights = deeplabv3.get_weights()
        # input_shape = (deeplabv3.inputs[0].shape[1], deeplabv3.inputs[0].shape[2], deeplabv3.inputs[0].shape[3])
        deeplabv3 = DeeplabV3Plus(deeplabv3.inputs[0].shape[1], deeplabv3.outputs[0].shape[-1], freeze_feature_extractor=True)
        deeplabv3.set_weights(weights)
    
    elif mode == 'freeze_bn':

        for layer in deeplabv3.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


    else:
        raise 'Configuração não conhecida'

    return deeplabv3


def freeze_segformer_layers(segformer, mode):

    if mode == 'unfreeze':
        segformer.trainable = True
        for layer in segformer.layers:
            layer.trainable = True
    elif mode == 'freeze_all':
        segformer.trainable = False
        for layer in segformer.layers:
            layer.trainable = False
    elif mode == 'freeze_encoder':
        for layer in segformer.layers:
            if layer.name.startswith('segdecoder'):
                break
            layer.trainable = False

    elif mode == 'freeze_bn':

        for layer in segformer.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    else:
        raise 'Configuração não conhecida'

    return segformer

def get_model(name, input_shape, num_classes=1):

    name = name.strip().lower()
    if name == 'segformerb0':
        return SegFormerB0(input_shape=input_shape, num_classes=num_classes)

    elif name == 'segformerb1':
        return SegFormerB1(input_shape=input_shape, num_classes=num_classes)
    
    elif name == 'segformerb2':
        return SegFormerB2(input_shape=input_shape, num_classes=num_classes)
    
    elif name == 'segformerb3':
        return SegFormerB3(input_shape=input_shape, num_classes=num_classes)
    
    elif name == 'segformerb4':
        return SegFormerB4(input_shape=input_shape, num_classes=num_classes)
    
    elif name == 'segsormerb5':
        return SegFormerB5(input_shape=input_shape, num_classes=num_classes)
    
    elif name == 'deeplabv3+':
        return DeeplabV3Plus(input_shape[0], num_classes=num_classes)
    
    elif name == 'unet':
        return get_unet(input_height=input_shape[0], input_width=input_shape[1], n_channels=input_shape[2])

    raise ValueError(f'The model {name} is not defined!')


def freeze_backbone_layers(backbone, mode):
    freeze_fn = None
    backbone_name = backbone.name.lower() 
    if backbone_name == 'unet' or backbone_name == 'u-net':
        freeze_fn = freeze_unet_layers
    elif backbone_name == 'deeplabv3+' or backbone_name == 'deeplabv3':
        freeze_fn = freeze_deeplabv3_layers
    elif backbone_name.startswith('segformer'):
        freeze_fn = freeze_segformer_layers

    if freeze_fn is None:
        raise ValueError(f'It was not possible to define how to freeze layers for {backbone_name}!')


    return freeze_fn(backbone, mode) 

def add_bn_at_start(backbone):
    shape = backbone.inputs[0].shape
    return tf.keras.Sequential([
        tf.keras.Input(shape=(shape[1], shape[2], shape[3])),
        BatchNormalization(),
        backbone,
    ])


def get_normalization_layer(quantification_values):
    normalization_layer = tf.keras.layers.Rescaling(1./quantification_values)
    return normalization_layer

