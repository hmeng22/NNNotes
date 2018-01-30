# TensorFlow

- **Tensor** consists of a set of primitive values shaped into an array of any number of dimensions.
- **Tensor's rank** is its number of dimensions.
- **Computational Graph** is a series of TensorFlow operations arranged into a graph of nodes.
- **Session** encapsulates the control and state of the TensorFlow runtime.
- **Placeholder** is a promise to provide a value later.
- **Variable** allows us to add trainable parameters to a graph.
- **Optimizers** the simplest optimizer is gradient descent.
- **Estimators** encapsulates training, evaluation, prediction, export.

```
Variables are the parameters of the algorithm and TensorFlow keeps track of how to change these to optimize the algorithm.

Placeholders are objects that allow you to feed in data of a specific type and shape.
```

```
init = tf.global_variables_initializer()
with tf.Session() as session:
  session.run(init)
```

```
with tf.name_scope('scope_name') as scope:
with tf.variable_scope("scope_name") as scope:
```

```
# Save & Restore

model_path = 'model/'
saver = tf.train.Saver()
save_path = saver.save(sess, model_path))
load_path = saver.restore(sess, model_path)


export_dir = 'saved_model/'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
builder.save()

tf.saved_model.loader.load()



# Tensorboard
# set names fro Variable & placeholder
# set tf.name_scope('scope_name')

summary_path = 'summary/'
tf.summary.histogram('conv1', conv1)
tf.summary.histogram('pool1', pool1)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter(summary_path, graph = tf.get_default_graph())
_, summary = sess.run([train_step, merged_summary_op], feed_dict = {})
summary_writer.add_summary(summary, epoch)


# CPU: '/cpu:0' ; GPU: '/gpu:0', '/gpu:1'
for i in range(2):
    with tf.device('/gpu:%d' % i):


# Graph
tf.Graph
    |- tf.Operation
    |- tf.Tensor
# associate a list of objects with a key
tf.GraphKeys
tf.add_to_collection
tf.get_collection

# Session
tf.ConfigProto
tf.RunOptions()
tf.RunMetadata()

g_1 = tf.Graph()
g_2 = tf.Graph()
with g_1.as_default():
    sess_1 = tf.Session(graph = g_1)

with g_2.as_default():
    sess_2 = tf.Session(graph = g_2)
```
