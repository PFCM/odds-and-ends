"""Try a TT-layer, compare with normal MLP"""
import os

import numpy as np
import tensorflow as tf

from ops import binary_connect

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
                      padding=None):
    if padding is None:
        padding = [[2, 2], [2, 2]]

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


def _tt_affine(input_var, input_modes, output_modes, tensor_ranks, scope=None,
               keep_prob=1.0):
    """Gets an affine transform where we represent the weight matrix in the
    tt-format. The below is mostly due to:
https://github.com/timgaripov/TensorNet-TF/blob/master/tensornet/layers/tt.py

    Args:
        input_var: input to the layer, can either be [batch_size, num_features]
            or [batch_size, *input_modes]. If the latter we reshape.
        input_modes: shape of the expected input tensors.
        output_modes: shape of expected output tensors.
        tensor_ranks: ranks of the decomposed matrix.
        scope: scope for operations.
        keep_prob: see if dropping out parts of the decomposition makes a
            difference.

    Returns:
        output of the layer [batch_size, prod(output_modes)].
    """
    with tf.variable_scope(scope or 'tt'):
        dim = len(input_modes)

        # not sure why cumsum, given later it's always being undone
        # mat_ps = np.cumsum(np.concatenate(
        #     ([0],
        #      tensor_ranks[:-1] * input_modes * output_modes *
        #      tensor_ranks[1:])))

        # mat_size = mat_ps[-1]
        # make a variable, for some reason by stitching together vectors?
        # no more of this outrageous behaviour
        cores = [tf.get_variable(
            'core_{}'.format(i), shape=[output_modes[i] * tensor_ranks[i + 1],
                                        tensor_ranks[i] * input_modes[i]])
                 for i in range(dim)]
        # for i in range(dim):
        #     n_in = tensor_ranks[i] * input_modes[i]
        #     mat_core = tf.truncated_normal([mat_ps[i+1] - mat_ps[i]],
        #                                    0.0,
        #                                    2.0 / n_in,
        #                                    tf.float32)
        #     if i == 0:
        #         mat = mat_core
        #     else:
        #         # seems like we really only need to do this once
        #         mat = tf.concat(0, [mat, mat_core])
        # # seems like a pretty long vector
        # mat = tf.get_variable('tt-mat', initializer=mat)

        out = tf.reshape(input_var, [-1, np.prod(input_modes)])
        out = tf.transpose(out, [1, 0])

        # now we can do the real tt-matrix product
        for i in range(dim):
            # this is where I start to lose it
            out = tf.reshape(out, [tensor_ranks[i] * input_modes[i], -1])

            # # so why did we bother to stick them all together
            # mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
            # # if this was what we wanted all along why didn't we just make a
            # # list of vars this shape?
            # mat_core = tf.reshape(
            #     mat_core, [tensor_ranks[i] * input_modes[i],
            #                output_modes[i] * tensor_ranks[i + 1]])
            # mat_core = tf.transpose(mat_core, [1, 0])
            mat_core = cores[i]

            # got the shapes we need, time for a product :)
            out = tf.matmul(mat_core, out)
            if keep_prob != 1.0:
                out = tf.nn.dropout(out, keep_prob)
            out = tf.reshape(out, [output_modes[i], -1])
            out = tf.transpose(out, [1, 0])
        biases = tf.get_variable('biases', shape=[np.prod(output_modes)])
        out = tf.add(tf.reshape(out, [-1, np.prod(output_modes)]), biases)
        return out


def get_mlp(num_hiddens=1024, nonlin=tf.nn.relu, scope=None):
    """Gets a standard feedforward model to compare"""
    with tf.variable_scope(scope or 'mlp') as scope:
        with tf.variable_scope('data'):
            train, test = get_mnist_tensors(
                FLAGS.batch_size, FLAGS.num_epochs, [1024])
            images, labels = train

        hiddens = nonlin(_affine(images, num_hiddens, 'input'), min_=0.0)
        # hiddens = tf.nn.dropout(hiddens, 0.5)
        logits = _affine(hiddens, 10, 'output')

        scope.reuse_variables()

        test_hiddens = nonlin(_affine(test[0], num_hiddens, 'input'), min_=0.0)
        test_logits = _affine(test_hiddens, 10, 'output')

    return logits, labels, test_logits, test[1]


def get_ttnet(inp_modes=None, inp_ranks=None, hidden_modes=None,
              hidden_ranks=None, out_modes=None, num_hiddens=1024,
              nonlin=tf.nn.relu, scope=None):
    """Gets a network with tensor layers"""
    with tf.variable_scope(scope or 'ttnet') as scope:
        if inp_modes is None:
            # inp_modes = np.array([4, 4, 4, 4, 4], dtype=np.int32)
            inp_modes = np.array([4] * 5, dtype=np.int32)
        if inp_ranks is None:  # ranks of first layer weights
            # inp_ranks = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
            inp_ranks = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        if hidden_modes is None:  # shape of hidden tensor
            # hidden_modes = np.array([4, 4, 4, 4, 4], dtype=np.int32)
            hidden_modes = np.array([500, 200], dtype=np.int32)
            hidden_modes_1 = np.array([10]*5, dtype=np.int32)
        if hidden_ranks is None:  # ranks of output tt-matrix
            hidden_ranks = np.array([1, 1, 1], dtype=np.int32)
        if out_modes is None:  # shape of output tensor
            output_modes = np.array([2, 5], dtype=np.int32)
        # get some inputs
        with tf.variable_scope('data'):
            train, test = get_mnist_tensors(
                FLAGS.batch_size, FLAGS.num_epochs, [1024])
            images, labels = train

        hiddens = nonlin(_tt_affine(
            images, inp_modes, hidden_modes_1, inp_ranks, scope='hidden_1',
            keep_prob=1.0))
        # hiddens = nonlin(_tt_affine(
        #     hiddens, inp_modes, hidden_modes_1, inp_ranks, scope='hidden_2',
        #     keep_prob=1.0))
        # hiddens = tf.nn.dropout(hiddens, 0.5)
        # logits = _affine(hiddens, 10, 'output')
        logits = _tt_affine(
            hiddens, hidden_modes, output_modes, hidden_ranks, scope='output')
        logits = tf.reshape(logits, [-1, 10])
        scope.reuse_variables()

        test_hiddens = nonlin(_tt_affine(
            test[0], inp_modes, hidden_modes_1, inp_ranks, scope='hidden_1'))
        # test_hiddens = nonlin(_tt_affine(
        #     test_hiddens, inp_modes, hidden_modes_1, inp_ranks,
        #     scope='hidden_2'))
        test_logits = _tt_affine(
            test_hiddens, hidden_modes, output_modes, hidden_ranks,
            scope='output')
        test_logits = tf.reshape(test_logits, [-1, 10])

        return logits, labels, test_logits, test[1]


def main(_):
    if FLAGS.model == 'mlp':
        train_out, train_labels, test_out, test_labels = get_mlp()
    elif FLAGS.model == 'tt':
        train_out, train_labels, test_out, test_labels = get_ttnet()
    elif FLAGS.model == 'bc':
        train_out, train_labels, test_out, test_labels = get_mlp(
            nonlin=binary_connect, num_hiddens=100)
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
    # all_summaries = tf.merge_all_summaries()
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    test_fname = os.path.join(FLAGS.results_dir, 'test.txt')
    # if it's alrady there, kill it (this lets us append as we go)
    if os.path.exists(test_fname):
        os.unlink(test_fname)
    summ_writer = tf.train.SummaryWriter(FLAGS.results_dir)  # , graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    test_steps = 10000 // FLAGS.batch_size

    try:
        step = 0
        while not coord.should_stop():
            if (step % 1000) == 0:
                test_acc = 0
                for _ in range(test_steps):
                    batch_acc = sess.run(test_accuracy)
                    test_acc += batch_acc
                test_acc /= test_steps
                print('\ntest accuracy: {}'.format(test_acc))
                with open(test_fname, 'a') as fp:
                    fp.write('{}, {}\n'.format(step, test_acc))

            batch_loss, _ = sess.run([loss_op, train_op])
            step += 1

            if (step % 100) == 0:
                print('\r({}) xent: {}'.format(step, batch_loss), end='')
                summaries = sess.run(tloss_summary)
                summ_writer.add_summary(summaries, global_step=step)

    except tf.errors.OutOfRangeError:
        coord.request_stop()
        print('ran out of data')

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
