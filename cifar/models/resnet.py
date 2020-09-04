"""
Resnet in Tensorflow.
reference: https://arxiv.org/pdf/1512.03385.pdf
"""

import tensorflow as tf


class Bottleneck(tf.keras.Model):
    """
        Bottleneck block of RESNET

        Args:
            in_filters (int): Number of channels in previous layer.
            filters(int): Number of filters in current block.
            stride (int): Stride of convolution layer.
    """

    expansion = 4

    def __init__(self, in_filters, filters, stride):
        super(Bottleneck, self).__init__()

        self.stride = stride
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters, kernel_size=(1, 1), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding='same',
                strides=stride,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                filters * self.expansion, kernel_size=(1, 1), use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])

        self.downsample = tf.keras.Sequential([])
        if in_filters != filters * self.expansion or stride != 1:
            self.downsample.add(
                tf.keras.layers.Conv2D(
                    self.expansion * filters,
                    kernel_size=(1, 1),
                    strides=stride,
                    use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, x, training):
        out = self.main(x, training=training)
        out += self.downsample(x, training=training)
        out = tf.nn.relu(out)
        return out


class BasicBlock(tf.keras.Model):
    """
        Basic block of RESNET

        Args:
            filters (int): Number of channels in current block.
            stride (int): Stride of convolution layer.
    """
    expansion = 1

    def __init__(self, _, filters, stride):
        super(BasicBlock, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                strides=stride,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                strides=1,
                use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.downsample = tf.keras.Sequential([])
        if stride != 1:
            self.downsample.add(
                tf.keras.layers.Conv2D(
                    filters, kernel_size=1, strides=stride, use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, x, training):

        out = self.main(x, training=training)
        out += self.downsample(x, training=training)
        out = tf.nn.relu(out)
        return out


class Resnet(tf.keras.Model):
    """
        RESNET architecture

        Args:
            block (Block): Basic or BottleNeck Block.
            number_blocks (tuple): Number stacked blocks with the same number of filters.
    """

    def __init__(self, block, number_blocks):
        super(Resnet, self).__init__()

        self.filters = [64, 128, 256, 512]
        self.in_filters = 64
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.in_filters,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        #block 64
        for _ in range(number_blocks[0]):
            self.main.add(block(self.in_filters, self.filters[0], stride=1))
            self.in_filters = self.filters[0] * block.expansion
        #block 128 - 512
        for i in range(1, len(self.filters)):
            self.main.add(block(self.in_filters, self.filters[i], stride=2))
            self.in_planes = self.filters[i] * block.expansion
            for _ in range(number_blocks[i] - 1):
                self.main.add(
                    block(self.in_filters, self.filters[i], stride=1))
                self.in_filters = self.filters[i] * block.expansion

        self.main.add(tf.keras.layers.AveragePooling2D(pool_size=4))
        self.main.add(tf.keras.layers.Flatten())
        self.main.add(tf.keras.layers.Dense(10))

    @tf.function
    def call(self, x, training):
        return self.main(x, training=training)


def ResNet18():
    """create ResNet18 architecture"""
    return Resnet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    """create ResNet50 architecture"""
    return Resnet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    """create ResNet101 architecture"""
    return Resnet(Bottleneck, [3, 4, 23, 3])
