"""Main file of different architecture implementations for Cifar."""

import os
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from models import mobilnet, regnet, resnet, vgg

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_float('wd', 1e-4, 'decoupled weight decay')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epochs', 200, 'batch straining epochs')
flags.DEFINE_string('save_dir', 'runs',
                    'Directory to save tensorboard logs to.')


def main(_):

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    now = datetime.now()
    hparam_str = "cifar" + now.strftime("%H:%M:%S") + "regnet"
    writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))
    writer.set_as_default()

    def augment(image, labels):
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.random_flip_left_right(image)
        paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
        image = tf.pad(image, paddings=paddings, constant_values=0)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image = (image - mean) / std
        return image, labels

    def preprocess_test(image, labels):
        image = tf.cast(image, tf.float32) / 255.
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image = (image - mean) / std
        return image, labels

    (train_ds, test_ds), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True)

    train_ds = train_ds.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            ds_info.splits['train'].num_examples,
            reshuffle_each_iteration=True).batch(FLAGS.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(
        preprocess_test,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            100).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    learning_rate = tf.Variable(FLAGS.lr)
    weight_decay = tf.Variable(FLAGS.wd)

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    with strategy.scope():

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        sparse_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        inputs = tf.keras.Input(shape=(32, 32, 3), name="imgs")
        outputs = regnet.RegNetX_200MF()(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tfa.optimizers.SGDW(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay)

        def compute_loss(labels, predictions):
            per_example_loss = sparse_loss(labels, predictions)
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=FLAGS.batch_size)

    def train_step(features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss

    def valid_step(features, labels):
        predictions = model(features, training=False)
        loss = compute_loss(labels, predictions)
        test_accuracy.update_state(labels, predictions)
        test_loss.update_state(loss)
        return loss

    train_ds = strategy.experimental_distribute_dataset(train_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

    @tf.function
    def distributed_train_step(dist_inputs, dist_labels):
        per_replica_losses = strategy.run(
            train_step, args=(dist_inputs, dist_labels))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_valid_step(dist_inputs, dist_labels):
        per_replica_losses = strategy.run(
            valid_step, args=(dist_inputs, dist_labels))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in tqdm(range(FLAGS.epochs)):

        train_accuracy.reset_states()
        train_loss = 0.0
        if epoch == 150:
            learning_rate.assign(learning_rate / 10)
            weight_decay.assign(weight_decay / 10)

        total_loss = 0.0
        num_batches = 0
        for imgs, labels in train_ds:
            total_loss += distributed_train_step(imgs, labels)
            num_batches += 1
        train_loss = total_loss / num_batches

        tf.summary.scalar("train/lr", learning_rate, step=epoch)
        tf.summary.scalar("train/losses", train_loss, step=epoch)
        tf.summary.scalar(
            "train/accuracy", train_accuracy.result(), step=epoch)

        test_accuracy.reset_states()
        test_loss.reset_states()
        for imgs, labels in test_ds:
            total_loss += distributed_valid_step(imgs, labels)

        tf.summary.scalar("test/accuracy", test_accuracy.result(), step=epoch)
        tf.summary.scalar("test/losses", test_loss.result(), step=epoch)
        tf.summary.flush()


if __name__ == '__main__':
    app.run(main)
