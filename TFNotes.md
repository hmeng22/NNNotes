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

tf.Graph
  |- tf.Operation
  |- tf.Tensor
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
