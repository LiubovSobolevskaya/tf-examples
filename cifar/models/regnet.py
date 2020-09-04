"""
Regnet in Tensorflow.
reference: https://arxiv.org/pdf/2003.13678.pdf
"""

import tensorflow as tf
import tensorflow_addons as tfa


class Block(tf.keras.Model):
    """
        Block of REGNET
        
        Args:
            block_in (int): Number of input channels.
            block_width (int): Number of output filters in current block.
            bottleneck_ratio (int): Determines number of filters in hidden layers
            group_width (int): Determines groups parameter of convolutional layer 
            stride (int): Stride of convolution layer.
    """

    def __init__(self, block_in, block_width, bottleneck_ratio, group_width,
                 stride):
        super(Block, self).__init__()

        self.stride = stride
        planes = int(block_width / bottleneck_ratio)

        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(planes, kernel_size=(1, 1), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        num_groups = planes // group_width

        self.main.add(
            tf.keras.layers.Conv2D(
                planes,
                kernel_size=(3, 3),
                strides=self.stride,
                padding='same',
                groups=num_groups,
                use_bias=False))
        self.main.add(tf.keras.layers.BatchNormalization())
        self.main.add(tf.keras.layers.Activation('relu'))
        self.main.add(
            tf.keras.layers.Conv2D(
                block_width, kernel_size=(1, 1), use_bias=False))
        self.main.add(tf.keras.layers.BatchNormalization())

        self.shortcut = tf.keras.Sequential([])
        if self.stride != 1 or block_in != block_width:
            self.shortcut.add(
                tf.keras.layers.Conv2D(
                    block_width,
                    kernel_size=(1, 1),
                    strides=self.stride,
                    use_bias=False))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training):
        out = self.main(inputs, training=training)
        out += self.shortcut(inputs, training=training)
        out = tf.nn.relu(out)

        return out


class Regnet(tf.keras.Model):
    """
        REGNET architecture 
    """

    def __init__(self, config):
        super(Regnet, self).__init__()
        self.config = config
        self.in_planes = 64
        self.bottleneck_ratio = 1

        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.in_planes,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        for i in range(len(self.config['depths'])):
            depth = self.config['depths'][i]
            width = self.config['widths'][i]
            group_width = self.config['group_width']
            strides = [self.config['strides'][i]
                       ] + [1] * (self.config['depths'][i] - 1)
            for j in range(depth):
                self.main.add(
                    Block(self.in_planes, width, self.bottleneck_ratio,
                          group_width, strides[j]))
                self.in_planes = width

        self.main.add(tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1)))
        self.main.add(tf.keras.layers.Flatten())
        self.main.add(tf.keras.layers.Dense(10))

    def call(self, inputs, training):
        out = self.main(inputs, training=training)

        return out


def RegNetX_200MF():
    """create RegNetX_200MF architecture"""
    config = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8
    }
    return Regnet(config)
