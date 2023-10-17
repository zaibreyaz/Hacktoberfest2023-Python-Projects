import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add

# Define the identity block
def identity_block(x, filters, block):
    F1, F2, F3 = filters
    
    x_shortcut = x
    
    # First component of main path
    x = Conv2D(F1, (1, 1), strides=(1, 1), padding='valid', name=f'conv_{block}_1')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_1')(x)
    x = Activation('relu')(x)
    
    # Second component of main path
    x = Conv2D(F2, (3, 3), strides=(1, 1), padding='same', name=f'conv_{block}_2')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_2')(x)
    x = Activation('relu')(x)
    
    # Third component of main path
    x = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=f'conv_{block}_3')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_3')(x)
    
    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

# Define the convolutional block
def convolutional_block(x, filters, block, stride=2):
    F1, F2, F3 = filters
    
    x_shortcut = x
    
    # First component of main path
    x = Conv2D(F1, (1, 1), strides=(stride, stride), padding='valid', name=f'conv_{block}_1')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_1')(x)
    x = Activation('relu')(x)
    
    # Second component of main path
    x = Conv2D(F2, (3, 3), strides=(1, 1), padding='same', name=f'conv_{block}_2')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_2')(x)
    x = Activation('relu')(x)
    
    # Third component of main path
    x = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=f'conv_{block}_3')(x)
    x = BatchNormalization(axis=3, name=f'bn_{block}_3')(x)
    
    # Shortcut path
    x_shortcut = Conv2D(F3, (1, 1), strides=(stride, stride), padding='valid', name=f'shortcut_{block}')(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=f'bn_shortcut_{block}')(x_shortcut)
    
    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

# Define the ResNet-50 model
def ResNet50(input_shape=(224, 224, 3), classes=1000):
    x_input = Input(input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = convolutional_block(x, [64, 64, 256], block='2a', stride=1)
    x = identity_block(x, [64, 64, 256], block='2b')
    x = identity_block(x, [64, 64, 256], block='2c')
    
    x = convolutional_block(x, [128, 128, 512], block='3a')
    x = identity_block(x, [128, 128, 512], block='3b')
    x = identity_block(x, [128, 128, 512], block='3c')
    x = identity_block(x, [128, 128, 512], block='3d')
    
    x = convolutional_block(x, [256, 256, 1024], block='4a')
    x = identity_block(x, [256, 256, 1024], block='4b')
    x = identity_block(x, [256, 256, 1024], block='4c')
    x = identity_block(x, [256, 256, 1024], block='4d')
    x = identity_block(x, [256, 256, 1024], block='4e')
    x = identity_block(x, [256, 256, 1024], block='4f')
    
    x = convolutional_block(x, [512, 512, 2048], block='5a')
    x = identity_block(x, [512, 512, 2048], block='5b')
    x = identity_block(x, [512, 512, 2048], block='5c')
    
    x = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(x)
    
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc')(x)
    
    model = tf.keras.Model(inputs=x_input, outputs=x, name='ResNet50')

    return model

# Create a ResNet-50 model
model = ResNet50()

# Display model summary
model.summary()
