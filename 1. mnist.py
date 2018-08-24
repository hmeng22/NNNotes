from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist.train (55000) & mnist.validation (5000) & mnist.test (10000)
# mnist.train.images & mnist.train.labels

import tensorflow as tf


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dnnModel(x):
    with tf.name_scope('reshape'):
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        o_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        o_pool1 = max_pool_2x2(o_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        o_conv2 = tf.nn.relu(conv2d(o_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        o_pool2 = max_pool_2x2(o_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        o_pool2_flat = tf.reshape(o_pool2, [-1, 7 * 7 * 64])
        o_fc1 = tf.nn.relu(tf.matmul(o_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y = tf.matmul(o_fc1_drop, W_fc2) + b_fc2

    return y, keep_prob


x = tf.placeholder(tf.float32, [None, 784])
y, keep_prob = dnnModel(x)

y_label = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y))

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_writer = tf.summary.FileWriter('GRAPH_data/')
train_writer.add_graph(tf.get_default_graph())

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('dropout_keep_prob', keep_prob)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50)

        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_label: batch_ys, keep_prob: 0.5})
        train_writer.add_summary(summary, epoch)

        if epoch % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_label: batch_ys, keep_prob: 1.0})
            validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels, keep_prob: 1.0})
            print('step %d, training accuracy %g, validation accuracy %g' % (epoch, train_accuracy, validation_accuracy))

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))

    train_writer.close()
