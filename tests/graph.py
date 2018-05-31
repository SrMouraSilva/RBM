from rbm.rbm import RBM
from rbm.util.util import prepare_graph
import tensorflow as tf

tf.set_random_seed(42)

# Model
rbm = RBM(visible_size=4, hidden_size=3)

# Visible layer
v = tf.placeholder(shape=[rbm.visible_size, None], name='v', dtype=tf.float32)

# Compile tensorflow methods
learn_op = rbm.learn(v)

# Learn?
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    writer = prepare_graph(session, '../graph/3')
    summary_op = tf.summary.merge_all()

    for i in range(10):
        if i % 50 == 0:
            print(i)

        visible = session.run(tf.random_uniform([rbm.visible_size, 1], minval=0, maxval=2, dtype=tf.int32))
        y = session.run(learn_op, feed_dict={v: visible})
        summary = session.run(summary_op, feed_dict={v: visible})

        writer.add_summary(summary, i)

    writer.close()
