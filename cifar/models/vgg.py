"""
VGG in Tensorflow.
reference: https://arxiv.org/pdf/1409.1556.pdf
"""

import tensorflow as tf


class VGG(tf.keras.Model):
    """
    VGG Architecture.
    Args:
        config (list): list of number of filters of Convolitional Layers and Max Polling layers.
    """

    def __init__(self, config):
        super(VGG, self).__init__()
        self.main = tf.keras.Sequential([])
        for layer in config:
            if layer == 'MP':
                self.main.add(
                    tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            else:
                self.main.add(
                    tf.keras.layers.Conv2D(
                        layer, kernel_size=(3, 3), padding='same'))
                self.main.add(tf.keras.layers.BatchNormalization())
                self.main.add(tf.keras.layers.Activation('relu'))

        self.main.add(tf.keras.layers.Flatten())
        self.main.add(tf.keras.layers.Dense(10))

    @tf.function
    def call(self, x, training):
        return self.main(x, training=training)


def VGG16():
    """create VGG16 architecture"""
    return VGG([
        64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP',
        512, 512, 512, 'MP'
    ])


def VGG19():
    """create VGG19 architecture"""
    return VGG([
        64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512,
        512, 'MP', 512, 512, 512, 512, 'MP'
    ])
