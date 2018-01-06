# I % 2 = O1*2 + O2
# I A B C O1 O2
# 0 0 0 0  0  0
# 1 0 0 1  0  1
# 2 0 1 0  1  0
# 3 0 1 1  1  1
# 4 1 0 0  2  0
# 5 1 0 1  2  1
# 6 1 1 0  3  0
# 7 1 1 1  3  1

import tensorflow as tf

train_x = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
train_y = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]]

X = tf.placeholder(tf.float32, [8, 3], name='InputData')
Y = tf.placeholder(tf.float32, [8, 2], name='LabelData')

weights_34 = tf.Variable(tf.random_normal([3, 4]), name='Weights_34')
weights_42 = tf.Variable(tf.random_normal([4, 2]), name='Weights_42')

biases_34 = tf.Variable(tf.random_normal([1, 4]), name='Biases_34')
biases_42 = tf.Variable(tf.random_normal([1, 2]), name='Biases_42')

with tf.name_scope('Layer_34'):
    layer_34 = tf.add(tf.matmul(X, weights_34), biases_34)
    layer_34_sigmoid = tf.nn.sigmoid(layer_34)

with tf.name_scope('Layer_42'):
    layer_42 = tf.add(tf.matmul(layer_34_sigmoid, weights_42), biases_42)
    layer_42_output = tf.nn.relu(layer_42)

with tf.name_scope('Loss'):
    cost = tf.square(Y - layer_42_output)

with tf.name_scope('GradientDescent'):
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

tf.summary.scalar('cost', tf.reduce_mean(cost))
merged_summary_op = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('/tmp/tb342', graph=tf.get_default_graph())

    for i in xrange(6000):
        _, summary = session.run([train, merged_summary_op], feed_dict={X: train_x, Y: train_y})
        summary_writer.add_summary(summary, i)

    print(session.run(layer_42_output, feed_dict={X: train_x}))
