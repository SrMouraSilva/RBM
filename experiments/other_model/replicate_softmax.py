# Copyright 2012 Hugo Larochelle, Stanislas Lauly. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle, Stanislas Lauly ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle, Stanislas Lauly OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle, Stanislas Lauly.

import tensorflow as tf


class ReplicatedSoftmax:
    """
    Replicated Softmax undirected topic model.

    Option ``n_stages`` is the number of training iterations.

    Options ``learning_rate`` and ``decrease_constant`` correspond
    to the learning rate and decrease constant used for stochastic
    gradient descent.

    Option ``hidden_size`` should be a positive integer specifying
    the number of hidden units (features).

    Option ``k_contrastive_divergence_steps`` is the number of
    Gibbs sampling iterations used by contrastive divergence.

    Option ``mean_field`` indicates that mean-field inference
    should be used to generate the negative statistics for learning.

    Option ``seed`` determines the seed for randomly initializing the
    weights.

    **Required metadata:**

    * ``'input_size'``:  Vocabulary size

    | **Reference:**
    | Replicated Softmax: an Undirected Topic Model
    | Salakhutdinov and Hinton
    | http://www.utstat.toronto.edu/~rsalakhu/papers/repsoft.pdf

    """
    def __init__(self, visible_size: int, hidden_size: int, n_stages, learning_rate=0.01, decrease_constant=0,
                 k_contrastive_divergence_steps=1, mean_field=False, seed=1234):

        self.visible_size = visible_size
        self.hidden_size = hidden_size

        self.n_stages = n_stages
        self.stage = 0
        self.learning_rate = learning_rate
        self.decrease_constant = decrease_constant
        self.k_contrastive_divergence_steps = k_contrastive_divergence_steps

        self.mean_field = mean_field
        self.seed = seed

        self.visible_size = 0

    def train(self, trainset):
        if self.stage == 0:
            self.initialize(trainset)
        for it in range(self.stage, self.n_stages):
            for example in trainset:
                self.update_learner(example)
        self.stage = self.n_stages

    def use(self, dataset):
        outputs = []
        for example in dataset:
            outputs += [self.use_learner(example)]
        return outputs

    def test(self, dataset):
        outputs = self.use(dataset)
        costs = []
        for example, output in itertools.izip(dataset, outputs):
            costs += [self.cost(output, example)]

        return outputs, costs

    def initialize(self, trainset):
        self.rng = np.random.mtrand.RandomState(self.seed)
        self.visible_size = trainset.metadata['input_size']

        self.W = tf.Variable(name='W', initial_value=2 * tf.random_normal([self.hidden_size, self.visible_size]) / self.visible_size,
                             dtype=tf.float32)

        self.b_h = tf.Variable(name='b_h', dtype=tf.float32, initial_value=tf.zeros([self.hidden_size, 1]))
        self.b_v = tf.Variable(name='b_v', dtype=tf.float32, initial_value=tf.ones([self.hidden_size, 1]) * 0.01)

        n = 0
        words = np.zeros((self.visible_size))
        for words_sparse in trainset:
            words[:] = 0
            words[words_sparse[1]] = words_sparse[0]
            self.b_v += words
            n += np.sum(words)
        self.b_v = np.log(self.b_v) - np.log(n)

        self.ΔW = np.zeros((self.hidden_size, self.visible_size))
        self.Δb_h = np.zeros((self.hidden_size))
        self.Δb_v = np.zeros((self.visible_size))

        self.input = np.zeros((self.visible_size))
        self.hidden = np.zeros((self.hidden_size))
        self.hidden_act = np.zeros((self.hidden_size))
        self.hidden_prob = np.zeros((self.hidden_size))

        self.neg_input = np.zeros((self.visible_size))
        self.neg_input_act = np.zeros((self.visible_size))
        self.neg_input_prob = np.zeros((self.visible_size))
        self.neg_hidden_act = np.zeros((self.hidden_size))
        self.neg_hidden_prob = np.zeros((self.hidden_size))

        self.neg_stats = np.zeros((self.hidden_size, self.visible_size))

        self.n_updates = 0

    def update_learner(self, example):
        η = self.learning_rate

        self.input[:] = 0
        self.input[example[1]] = example[0]
        n_words = int(self.input.sum())

        # Performing CD-k
        mllin.product_matrix_vector(self.W, self.input, self.hidden_act)
        self.hidden_act += self.b_h * n_words
        mlnonlin.sigmoid(self.hidden_act, self.hidden_prob)
        self.neg_hidden_prob[:] = self.hidden_prob

        for k in range(self.k_contrastive_divergence_steps):
            if self.mean_field:
                self.hidden[:] = self.neg_hidden_prob
            else:
                np.less(self.rng.rand(self.hidden_size), self.neg_hidden_prob, self.hidden)

            mllin.product_matrix_vector(self.W.T, self.hidden, self.neg_input_act)
            self.neg_input_act += self.b_v
            mlnonlin.softmax(self.neg_input_act, self.neg_input_prob)
            if self.mean_field:
                self.neg_input[:] = n_words * self.neg_input_prob
            else:
                self.neg_input[:] = self.rng.multinomial(n_words, self.neg_input_prob)

            mllin.product_matrix_vector(self.W, self.neg_input, self.neg_hidden_act)
            self.neg_hidden_act += self.b_h * n_words
            mlnonlin.sigmoid(self.neg_hidden_act, self.neg_hidden_prob)

        mllin.outer(self.hidden_prob, self.input, self.ΔW)
        mllin.outer(self.neg_hidden_prob, self.neg_input, self.neg_stats)
        self.ΔW -= self.neg_stats

        np.subtract(self.input, self.neg_input, self.Δb_v)
        np.subtract(self.hidden_prob, self.neg_hidden_prob, self.Δb_h)

        self.ΔW *= η / (1. + self.decrease_constant * self.n_updates)
        self.Δb_v *= η / (1. + self.decrease_constant * self.n_updates)
        self.Δb_h *= n_words * η / (1. + self.decrease_constant * self.n_updates)

        self.W += self.ΔW
        self.b_v += self.Δb_v
        self.b_h += self.Δb_h

        self.n_updates += 1

    def use_learner(self, example):
        self.input[:] = 0
        self.input[example[1]] = example[0]
        output = np.zeros((self.hidden_size))
        mllin.product_matrix_vector(self.W, self.input, self.hidden_act)
        self.hidden_act += self.b_h * self.input.sum()
        mlnonlin.sigmoid(self.hidden_act, output)

        return [output]

    def compute_document_representation(self, word_counts_sparse):
        self.input[:] = 0
        self.input[word_counts_sparse[1]] = word_counts_sparse[0]
        output = np.zeros((self.hidden_size,))
        mllin.product_matrix_vector(self.W, self.input, self.hidden_act)
        self.hidden_act += self.b_h * self.input.sum()
        mlnonlin.sigmoid(self.hidden_act, output)
        return output

    def cost(self, outputs, example):
        hidden = outputs[0]
        self.input[:] = 0
        self.input[example[1]] = example[0]
        mllin.product_matrix_vector(self.W.T, hidden, self.neg_input_act)
        self.neg_input_act += self.b_v
        mlnonlin.softmax(self.neg_input_act, self.neg_input_prob)

        return [np.sum((self.input - self.input.sum() * self.neg_input_prob) ** 2)]
