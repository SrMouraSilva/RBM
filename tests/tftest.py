import tensorflow as tf

a = tf.constant(2, name="a")
b = 3
c = tf.add(a, b, name='Add')
print(c)

with tf.Session() as session:
    print(session.run(c))

import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Useless')

with tf.Session() as session:
    pow_out = session.run(pow_op)
    print(pow_op)
    useless_out = session.run(useless_op)
    print(useless_out)

h = tf.constant([1, 2, 3], name='h')
v = m = tf.constant([1, 1, 0, 1], name='v')

tf.Tensor.T = property(lambda self: tf.transpose(self))

def outer(x, y):
    return tf.einsum('i,j->ij', x, y)


with tf.Session() as session:
    #print(b_v.T)
    #print(session.run(v @ b_v.T))
    print(session.run(outer(h, v)))
    print(session.run(h.T @ v))
