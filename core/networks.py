from keras.layers import Input, Activation, Add, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Concatenate, Multiply
from keras.layers import GlobalAveragePooling2D, Reshape, Permute


from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


from keras.layers import MaxPooling2D
from keras.layers import Dropout, concatenate


img_size = 512

image_shape = (img_size, img_size, 4)
image_d_shape = (img_size, img_size, 3)



def convolution_2d(x, num_filter=32, k_size=3, act_type="mish"):
            
    x = Conv2D(num_filter, k_size, padding='same', kernel_initializer = 'he_normal')(x)
    # x = BatchNormalization()(x)
    
    if act_type=="mish": 
        softplus_x = Activation('softplus')(x)
        tanh_softplus_x = Activation('tanh')(softplus_x)
        x = multiply([x, tanh_softplus_x])

    elif act_type=="swish":
        sigmoid_x = Activation('sigmoid')(x)
        x = multiply([x, sigmoid_x])
        
    elif act_type=="leakyrelu": x = LeakyReLU(alpha=0.1)(x)
    elif act_type=="tanh": x = Activation('tanh')(x)
    
    return x


def gan_model(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model


### EDN-GTM-L
def unet_spp_large_swish_generator_model():
       
    inputs = Input(image_shape)
    
    conv1 = convolution_2d(inputs, 64, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 64, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 64, 3,  act_type="swish")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_2d(pool1, 128, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 128, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 128, 3,  act_type="swish")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = convolution_2d(pool2, 256, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 256, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 256, 3,  act_type="swish")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = convolution_2d(pool3, 512, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 512, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 512, 3,  act_type="swish")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = convolution_2d(pool4, 1024, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 1024, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 1024, 3,  act_type="swish")
    
    # SPP #
    conv5 = convolution_2d(conv5, 512, 1, act_type="swish")
    
    conv5 = concatenate([conv5,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5)], axis = 3)

    conv5 = convolution_2d(conv5, 1024, 1, act_type="swish")
    drop5 = Dropout(0.5)(conv5)

    up6 = convolution_2d((UpSampling2D(size = (2,2))(drop5)), 512, 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = convolution_2d(merge6, 512, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 512, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 512, 3,  act_type="swish")
    
    up7 = convolution_2d((UpSampling2D(size = (2,2))(conv6)), 256, 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = convolution_2d(merge7, 256, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 256, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 256, 3,  act_type="swish")
   
    up8 = convolution_2d((UpSampling2D(size = (2,2))(conv7)), 128, 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = convolution_2d(merge8, 128, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 128, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 128, 3,  act_type="swish")

    up9 = convolution_2d((UpSampling2D(size = (2,2))(conv8)), 64, 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = convolution_2d(merge9, 64, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 64, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 64, 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


### EDN-GTM-B
def unet_spp_base_swish_generator_model():
       
    inputs = Input(image_shape)
    
    conv1 = convolution_2d(inputs, 64, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 64, 3,  act_type="swish")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_2d(pool1, 128, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 128, 3,  act_type="swish")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = convolution_2d(pool2, 256, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 256, 3,  act_type="swish")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = convolution_2d(pool3, 512, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 512, 3,  act_type="swish")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = convolution_2d(pool4, 1024, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 1024, 3,  act_type="swish")
    
    # SPP #
    conv5 = convolution_2d(conv5, 512, 1, act_type="swish")
    
    conv5 = concatenate([conv5,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5)], axis = 3)

    conv5 = convolution_2d(conv5, 1024, 1, act_type="swish")
    drop5 = Dropout(0.5)(conv5)

    up6 = convolution_2d((UpSampling2D(size = (2,2))(drop5)), 512, 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = convolution_2d(merge6, 512, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 512, 3,  act_type="swish")
    
    up7 = convolution_2d((UpSampling2D(size = (2,2))(conv6)), 256, 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = convolution_2d(merge7, 256, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 256, 3,  act_type="swish")
   
    up8 = convolution_2d((UpSampling2D(size = (2,2))(conv7)), 128, 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = convolution_2d(merge8, 128, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 128, 3,  act_type="swish")

    up9 = convolution_2d((UpSampling2D(size = (2,2))(conv8)), 64, 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = convolution_2d(merge9, 64, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 64, 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


### EDN-GTM-S
def unet_spp_small_swish_generator_model():
       
    inputs = Input(image_shape)
    
    conv1 = convolution_2d(inputs, 32, 3,  act_type="swish")
    conv1 = convolution_2d(conv1, 32, 3,  act_type="swish")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_2d(pool1, 64, 3,  act_type="swish")
    conv2 = convolution_2d(conv2, 64, 3,  act_type="swish")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = convolution_2d(pool2, 128, 3,  act_type="swish")
    conv3 = convolution_2d(conv3, 128, 3,  act_type="swish")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = convolution_2d(pool3, 256, 3,  act_type="swish")
    conv4 = convolution_2d(conv4, 256, 3,  act_type="swish")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = convolution_2d(pool4, 512, 3,  act_type="swish")
    conv5 = convolution_2d(conv5, 512, 3,  act_type="swish")
    
    # SPP #
    conv5 = convolution_2d(conv5, 256, 1, act_type="swish")
    
    conv5 = concatenate([conv5,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5)], axis = 3)

    conv5 = convolution_2d(conv5, 512, 1, act_type="swish")
    drop5 = Dropout(0.5)(conv5)

    up6 = convolution_2d((UpSampling2D(size = (2,2))(drop5)), 256, 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = convolution_2d(merge6, 256, 3,  act_type="swish")
    conv6 = convolution_2d(conv6, 256, 3,  act_type="swish")
    
    up7 = convolution_2d((UpSampling2D(size = (2,2))(conv6)), 128, 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = convolution_2d(merge7, 128, 3,  act_type="swish")
    conv7 = convolution_2d(conv7, 128, 3,  act_type="swish")
   
    up8 = convolution_2d((UpSampling2D(size = (2,2))(conv7)), 64, 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = convolution_2d(merge8, 64, 3,  act_type="swish")
    conv8 = convolution_2d(conv8, 64, 3,  act_type="swish")

    up9 = convolution_2d((UpSampling2D(size = (2,2))(conv8)), 32, 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = convolution_2d(merge9, 32, 3,  act_type="swish")
    conv9 = convolution_2d(conv9, 32, 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


## Critic
def unet_encoder_discriminator_model():
    
    inputs = Input(shape=image_d_shape)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    x = GlobalAveragePooling2D()(drop5)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model
