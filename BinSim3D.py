import numpy as np
from tensorflow.keras import layers, Model, Input
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
from keras.utils import plot_model
from tqdm import tqdm
from tensorflow.keras.metrics import Precision, Recall, AUC


def cbam_block_3d(input_feature, ratio=8):
    channel = layers.GlobalAveragePooling3D()(input_feature)
    channel = layers.Dense(input_feature.shape[-1] // ratio, activation='relu')(channel)
    channel = layers.Dense(input_feature.shape[-1], activation='sigmoid')(channel)
    channel_attention = layers.Multiply()([input_feature, channel])

    spatial_avg = tf.reduce_mean(channel_attention, axis=-1, keepdims=True)
    spatial_max = tf.reduce_max(channel_attention, axis=-1, keepdims=True)
    spatial = layers.Concatenate(axis=-1)([spatial_avg, spatial_max])
    spatial = layers.Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(spatial)
    return layers.Multiply()([channel_attention, spatial])


def non_local_block_3d(input_tensor, compression_ratio=2):
    batch_size = tf.shape(input_tensor)[0]
    t, h, w, c = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]
    reduced_channels = c // compression_ratio

    theta = layers.Reshape((t * h * w, c))(input_tensor)
    theta = layers.Dense(reduced_channels)(theta)

    phi = layers.Reshape((t * h * w, c))(input_tensor)
    phi = layers.Dense(reduced_channels)(phi)
    phi = layers.Permute((2, 1))(phi)

    attn = layers.Dot(axes=(2, 1))([theta, phi])
    attn = layers.Activation('softmax')(attn)

    g = layers.Reshape((t * h * w, c))(input_tensor)
    g = layers.Dense(reduced_channels)(g)

    out = layers.Dot(axes=(2, 1))([attn, g])
    out = layers.Reshape((t, h, w, reduced_channels))(out)
    out = layers.Dense(c)(out)

    return layers.Add()([input_tensor, out])

class SiameseVideoModel:
    def __init__(self, frame_count=50, height=64, width=128, channels=1):
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.channels = channels
        self.input_shape = (frame_count, height, width, channels)
        self.model = self.build_model()

    def build_base_network(self):
        input = Input(shape=self.input_shape)

        x = layers.Conv3D(32, (3, 5, 5), activation='relu', padding='same')(input)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.3)(x)

        # 全连接层
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)

        return Model(input, x)

    def build_model(self):
        base_network = self.build_base_network()

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
        dense1 = layers.Dense(512, activation='relu')(l1_distance)
        output = layers.Dense(2, activation='softmax')(dense1)

        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        plot_model(model, show_shapes=True,to_file='model.png')
        return model


def main():
    model = SiameseVideoModel(frame_count=50, height=256, width=256, channels=1)

if __name__ == "__main__":
    main()