import os

import numpy as np
import tensorflow as tf

from rnndatasets import sequentialmnist as sm


flags = tf.app.flags
flags.DEFINE_string('model', 'mlp', 'whether to use mlp or tt')
flags.DEFINE_string('results_dir', '.', 'where to put summaries')

flags.DEFINE_integer('batch_size', 50, 'how many')
flags.DEFINE_integer('num_epochs', 100, 'how long')
flags.DEFINE_float('learning_rate', 0.01, 'step size')
FLAGS = flags.FLAGS


def get_batches(dataset, num_images, batch_size, num_epochs, padd):
    images, labels = sm.get_data(dataset, num_images)
    image, label = tf.train.slice_input_producer(
        [tf.constant(images.reshape((-1, 28, 28))),
         tf.constant(labels.astype(np.int32))],
        shuffle=True, num_epochs=num_epochs)
    image = tf.pad(image, padd)
    return tf.train.batch([image, label], batch_size=batch_size)


def get_mnist_tensors(batch_size, num_epochs,
                      final_shape,
                      padding=[[2, 2], [2, 2]]):
    train_image_batch, train_label_batch = get_batches(
        'train', 60000, batch_size, num_epochs, padding)
    test_image_batch, test_label_batch = get_batches(
        'test', 10000, batch_size, None, padding)

    new_shape = [-1] + final_shape

    train_image_batch = tf.reshape(train_image_batch, new_shape)
    test_image_batch = tf.reshape(test_image_batch, new_shape)

    return ((train_image_batch, train_label_batch),
            (test_image_batch, test_label_batch))


def count_params():
    total = 0
    for var in tf.trainable_variables():
        prod = 1
        for dim in var.get_shape().as_list():
            prod *= dim
        total += prod
    return total


def _affine(input_var, new_size, scope=None):
    with tf.variable_scope(scope or 'affine'):
        input_size = input_var.get_shape()[1].value
        weights = tf.get_variable('weights', [input_size, new_size])
        bias = tf.get_variable('bias', [new_size])

        return tf.nn.bias_add(tf.matmul(input_var, weights), bias)


def get_mlp(num_hiddens=1024, nonlin=tf.nn.relu, scope=None):
    """Gets a standard feedforward model to compare"""
    with tf.variable_scope(scope or 'mlp') as scope:
        with tf.variable_scope('data'):
            train, test = get_mnist_tensors(
                FLAGS.batch_size, FLAGS.num_epochs, [1024])
            images, labels = train

        hiddens = nonlin(_affine(images, num_hiddens, 'input'))
        logits = _affine(hiddens, 10, 'output')

        scope.reuse_variables()

        test_hiddens = nonlin(_affine(test[0], num_hiddens, 'input'))
        test_logits = _affine(test_hiddens, 10, 'output')

    return logits, labels, test_logits, test[1]


def get_ttnet():
    """Gets a network with tensor layers"""
    pass


def main(_):
    if FLAGS.model == 'mlp':
        train_out, train_labels, test_out, test_labels = get_mlp()
    else:
        raise NotImplementedError('have not done {}'.format(FLAGS.model))

    print('{:*^50}'.format(
        'Got model with {} parameters'.format(count_params())))

    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        train_out, train_labels)
    loss_op = tf.reduce_mean(loss_op)
    tloss_summary = tf.scalar_summary('loss', loss_op)

    test_accuracy = tf.contrib.metrics.accuracy(
        tf.cast(tf.argmax(test_out, 1), tf.int32),
        test_labels)
    tf.scalar_summary('test_accuracy', test_accuracy)

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_op)

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    print('{:*^50}'.format('initialised'))

    # get summaries
    all_summaries = tf.merge_all_summaries()
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    summ_writer = tf.train.SummaryWriter(FLAGS.results_dir, graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    test_steps = 10000 // FLAGS.batch_size

    try:
        step = 0
        while not coord.should_stop():
            batch_loss, _ = sess.run([loss_op, train_op])
            step += 1

            if (step % 100) == 0:
                print('\r({}) xent: {}'.format(step, batch_loss), end='')
                summaries = sess.run(tloss_summary)
                summ_writer.add_summary(summaries, global_step=step)
            if (step % 1000) == 0:
                test_acc = 0
                for _ in range(test_steps):
                    batch_acc = sess.run(test_accuracy)
                    test_acc += batch_acc
                test_acc /= test_steps
                print('\ntest accuracy: {}'.format(test_acc))

    except tf.errors.OutOfRangeError:
        print('ran out of data')


if __name__ == '__main__':
    tf.app.run()
