from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def safe_pool(x, pool_size=(2, 2), name=None):
    h, w = x.shape[1], x.shape[2]
    pool_h = pool_size[0] if h is not None and h >= pool_size[0] else 1
    pool_w = pool_size[1] if w is not None and w >= pool_size[1] else 1
    return layers.MaxPooling2D(pool_size=(pool_h, pool_w), name=name)(x)


def feature_cnn_block(input_shape):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (5, 5), strides=(2, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = safe_pool(x)
    x = layers.Dropout(0.11)(x)

    x = layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = safe_pool(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(96, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = safe_pool(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(96, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = safe_pool(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Reshape((-1, x.shape[-1]))(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=x.shape[-1])(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)

    return models.Model(input_layer, x)


def MultiFeatureCNN(input_shapes):
    cnn_models = [feature_cnn_block(shape) for shape in input_shapes]
    merged_train = tf.keras.layers.Concatenate()(
        [model.output for model in cnn_models]
    )
    x = tf.keras.layers.Dense(128, activation='relu')(merged_train)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[m.input for m in cnn_models], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
