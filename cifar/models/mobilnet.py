"""
Mobilnet in Tensorflow
reference: https://arxiv.org/pdf/1801.04381.pdf
"""
import tensorflow as tf


class Block(tf.keras.Model):
    """
        Block of Mobilnet

        Args:
            in_filters (int): Number of input channels.
            out_filters(int): Number of output filters in current block.
            expantion(int): Determines number of filters in hidden layers.
            stride (int): Stride of convolution layer.
    """

    def __init__(self, in_filters, out_filters, expantion, stride):
        super(Block, self).__init__()
        hidden_planes = in_filters * expantion
        self.stride = stride
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                hidden_planes,
                kernel_size=(1, 1),
                padding='valid',
                strides=(1, 1),
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                hidden_planes,
                kernel_size=(3, 3),
                padding='same',
                strides=(self.stride, self.stride),
                use_bias=False,
                groups=hidden_planes),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                out_filters,
                kernel_size=(1, 1),
                padding='valid',
                strides=(1, 1),
                use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut = None
        if self.stride == 1 and in_filters != out_filters:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    out_filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='valid',
                    use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    @tf.function
    def call(self, x, training):
        out = self.main(x, training=training)
        if self.shortcut is not None:
            out += self.shortcut(x, training=training)
        elif self.stride == 1:
            out += x
        return out


class MobileNetV2(tf.keras.Model):

    config = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
              (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self):
        super(MobileNetV2, self).__init__()
        """
        MobileNetV2
        """
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                padding='same',
                strides=(1, 1),
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        in_planes = 32
        for expansion, out_planes, num_blocks, strd in self.config:
            strides = [strd] + [1] * (num_blocks - 1)
            for stride in strides:
                self.main.add(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes

        self.main.add(
            tf.keras.layers.Conv2D(
                1280,
                kernel_size=1,
                strides=(1, 1),
                padding='valid',
                use_bias=False))
        self.main.add(tf.keras.layers.BatchNormalization())
        self.main.add(tf.keras.layers.Activation('relu'))
        self.main.add(tf.keras.layers.AveragePooling2D(pool_size=4))
        self.main.add(tf.keras.layers.Flatten())
        self.main.add(tf.keras.layers.Dense(10))

    @tf.function
    def call(self, x, training):
        out = self.main(x, training=training)
        return out
