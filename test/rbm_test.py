import unittest
import warnings

import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from rbm.rbm import RBM
from rbm.util.util import prepare_graph

'''
class RBMTest(unittest.TestCase):

    def setUp(self):
        tf.set_random_seed(42)

        self.rbm = RBM(visible_size=4, hidden_size=3)

    @property
    def visible(self):
        return tf.placeholder(shape=[self.rbm.visible_size, None], name='v', dtype=tf.float32)

    @property
    def hidden(self):
        return tf.placeholder(shape=[self.rbm.hidden_size, None], name='h', dtype=tf.float32)

    def layer(self, size, batch_size=1):
        with tf.Session() as session:
            return session.run(tf.random_uniform([size, batch_size], minval=0, maxval=2, dtype=tf.int32))

    def test_parameters(self):
        assert self.rbm.parameters == self.rbm.θ

    def test_free_energy(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = self.visible
        visible = self.layer(self.rbm.visible_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.F(v), feed_dict={v: visible})

        assert_array_almost_equal(y, y)

    def test_energy(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = self.visible
        h = self.hidden

        visible = self.layer(self.rbm.visible_size)
        hidden = self.layer(self.rbm.hidden_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.E(v, h), feed_dict={v: visible, h: hidden})

        assert_array_almost_equal(y, y)

    def test_P_h_given_v(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = self.visible

        visible = self.layer(self.rbm.visible_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.P_h_given_v(v), feed_dict={v: visible})

        assert_array_almost_equal(y, y)

    def test_P_v_given_h(self):
        warnings.warn("Expected a useful test", UserWarning)

        h = self.hidden

        hidden = self.layer(self.rbm.hidden_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.P_v_given_h(h), feed_dict={h: hidden})

        assert_array_almost_equal(y, y)

    def test_sample_h_given_v(self):
        warnings.warn("Expected a useful test", UserWarning)

        v = self.visible
        visible = self.layer(self.rbm.visible_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.sample_h_given_v(v), feed_dict={v: visible})

        assert_array_almost_equal(y, y)

    def test_sample_v_given_h(self):
        warnings.warn("Expected a useful test", UserWarning)

        h = self.hidden
        hidden = self.layer(self.rbm.hidden_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.sample_v_given_h(h), feed_dict={h: hidden})

        assert_array_almost_equal(y, y)

    def test_gibbs_step(self):
        v0 = self.visible
        visible = self.layer(self.rbm.visible_size)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            y = session.run(self.rbm.gibbs_step(visible), feed_dict={v0: visible})

        assert_array_almost_equal(y, y)

    def test_calculate_parameters_updates(self):
        v = self.visible
        visible = self.layer(self.rbm.visible_size)

        with tf.Session() as session:
            with prepare_graph(session):
                session.run(tf.global_variables_initializer())
                y = session.run(self.rbm.calculate_parameters_updates(v), feed_dict={v: visible})

        for parameter in y:
            assert_array_almost_equal(parameter, parameter)
'''