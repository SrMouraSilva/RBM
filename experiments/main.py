from rbm.util.util import softmax, Σ

import tensorflow as tf

session = tf.InteractiveSession()

x = tf.constant([
    [1., 2., 3.],
    [2., 2., 2.],
    [1., 1., 4.],
])
y = softmax(x)
print(y.eval())

#print(Σ(y, axis=1).eval())

sample_uniform = tf.random_uniform(tf.shape(x))
y = y - sample_uniform
print(y.eval())

y = tf.nn.relu(tf.sign(x - sample_uniform))
print(y.eval())
#y = tf.nn.relu(tf.sign(x - sample_uniform))
#print(y.eval())
