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
import numpy as np
from numpy.random.mtrand import RandomState

from experiments.other_model.activation_function.activation_function import Softmax, ActivationFunction
from experiments.other_model.topic_modeling import TopicModel


class DocNADE(TopicModel):
    """
    Neural autoregressive distribution estimator for topic model.

    Option ``n_stages`` is the number of training iterations.

    Option ``hidden_size`` should be a positive integer specifying
    the number of hidden units (features).

    Options ``learning_rate`` is the learning rate (default=0.001).

    Option ``activation_function`` is a implementation of ActivationFunction.

    Option ``normalize_by_document_size`` normalize the learning by
    the size of the documents.

    Option ``hidden_bias_scaled_by_document_size`` scale the bias
    of the hidden units by the document size.

    Option ``seed`` determines the seed for randomly initializing the
    weights.

    **Required metadata:**

    * ``'input_size'``:  Vocabulary size

    | **Reference:**
    | A Neural Autoregressive Topic Model
    | Larochelle and Lauly
    | http://www.dmi.usherb.ca/~larocheh/publications/docnade.pdf
    """

    def __init__(self, vocabulary_size, hidden_size, n_stages=10, learning_rate=0.001, activation_function: ActivationFunction=None,
                 testing_ensemble_size=1, normalize_by_document_size=True, hidden_bias_scaled_by_document_size=False, seed=1234):
        self.n_stages = n_stages

        self.learning_rate = learning_rate
        self.activation_function = activation_function if activation_function is not None else Softmax
        self.hidden_size = hidden_size
        self.testing_ensemble_size = testing_ensemble_size
        self.normalize_by_document_size = normalize_by_document_size
        self.hidden_bias_scaled_by_document_size = hidden_bias_scaled_by_document_size

        self.vocabulary_size = vocabulary_size

        # Weights and bias
        self.W = self.rng.rand(self.vocabulary_size, self.hidden_size) / (self.vocabulary_size * self.hidden_size)
        self.V = self.rng.rand(self.vocabulary_size, self.hidden_size) / (self.vocabulary_size * self.hidden_size)
        # self.V = np.zeros((self.voc_size,self.hidden_size))

        self.c = np.zeros((self.hidden_size,))

        self.dV = np.zeros((self.vocabulary_size, self.hidden_size))
        self.dW = np.zeros((self.vocabulary_size, self.hidden_size))
        self.db = np.zeros((self.vocabulary_size,))
        self.dc = np.zeros((self.hidden_size,))

        self.rng = RandomState(seed)

    def initialize(self, trainset):
        # Create word tree
        def get_binary_codes_rec(beg, end, depth, binary_codes, node_ids, path_lengths):
            if end - beg == 1:
                path_lengths[beg] = depth
                return

            i, j, k = beg, beg + np.floor((end - beg)/2), end
            binary_codes[i:j, depth] = True
            binary_codes[j:k, depth] = False
            node_ids[i:k, depth] = j
            get_binary_codes_rec(i, j, depth+1, binary_codes, node_ids, path_lengths)
            get_binary_codes_rec(j, k, depth+1, binary_codes, node_ids, path_lengths)

        def get_binary_codes(vocabulary_size):
            tree_depth = np.int32(np.ceil(np.log2(vocabulary_size)))
            binary_codes = np.zeros((vocabulary_size, tree_depth), dtype=bool)
            node_ids = np.zeros((vocabulary_size, tree_depth), dtype=np.int32)
            path_lengths = np.zeros((vocabulary_size), dtype=np.int32)
            get_binary_codes_rec(0, vocabulary_size, 0, binary_codes, node_ids, path_lengths)
            return binary_codes, path_lengths, node_ids

        self.binary_codes, self.path_lengths, self.node_ids = get_binary_codes(self.vocabulary_size)
        self.tree_depth = self.binary_codes.shape[1]

        # Initialize b to base rate
        freq_pos = np.ones((self.vocabulary_size,)) * 0.01
        freq_neg = np.ones((self.vocabulary_size,)) * 0.01
        cnt = 0
        for word_counts, word_ids in trainset:
            cnt += 1
            if cnt > 50000:
                break
            for i in range(self.tree_depth):
                # Node at level i
                for k in range(len(word_ids)):
                    n = self.node_ids[word_ids[k], i]
                    c = self.binary_codes[word_ids[k], i]
                    if c:
                        # Nb. of left decisions
                        freq_pos[n] += word_counts[k]
                    else:
                        # Nb. of right decisions
                        freq_neg[n] += word_counts[k]

        p = freq_pos / (freq_pos + freq_neg)
        # Convert marginal probabilities to sigmoid bias
        self.b = - np.log(1 / p - 1)

        # ...
        self.to_update = np.zeros((self.vocabulary_size,), dtype=bool)

    def compute_document_representation(self, word_counts_sparse):
        new_inp = np.zeros((self.vocabulary_size,))
        new_inp[word_counts_sparse[1]] = word_counts_sparse[0]
        out = np.zeros((1, self.hidden_size))
        out = self.activation_function((self.c + np.dot(new_inp, self.W)).reshape((1, -1)))
        return out.reshape((-1,))

    def fprop_word_probs(self):
        """
        Computes the tree decision probs for all p(w_i | w_i<).
        """
        self.probs = np.ones((self.h.shape[0], self.tree_depth))
        tm_utils.doc_mnade_tree_fprop_word_probs(self.h, self.V, self.b, self.words, self.binary_codes,
                                                 self.path_lengths,
                                                 self.node_ids, self.probs)

    def bprop_word_probs(self):
        """
        Computes db, dV and dh
        """
        self.dh = np.zeros(self.h.shape)
        tm_utils.doc_mnade_tree_bprop_word_probs(self.h, self.dh, self.V, self.dV, self.b, self.db, self.words,
                                                 self.binary_codes,
                                                 self.path_lengths, self.node_ids, self.probs, self.to_update)

    def update_word_probs(self):
        """
        Updates b and V
        """
        if self.normalize_by_document_size:
            tm_utils.doc_mnade_tree_update_word_probs(self.V, self.dV, self.b, self.db, self.words,
                                                      self.path_lengths, self.node_ids, self.to_update,
                                                      self.learning_rate)
        else:
            tm_utils.doc_mnade_tree_update_word_probs(self.V, self.dV, self.b, self.db, self.words,
                                                      self.path_lengths, self.node_ids, self.to_update,
                                                      self.learning_rate * len(self.words))

    def fprop(self, word_ids, word_counts):
        self.word_ids = word_ids
        self.words = tm_utils.words_list_from_counts(word_ids, word_counts)
        self.rng.shuffle(self.words)
        self.act = np.zeros((len(self.words), self.hidden_size))
        np.add.accumulate(self.W[self.words[:-1], :], axis=0, out=self.act[1:, :])
        if self.hidden_bias_scaled_by_document_size:
            self.act += self.c * len(self.words)
        else:
            self.act += self.c
        self.h = np.zeros((len(self.words), self.hidden_size))
        self.h = self.activation_function(self.act)
        self.fprop_word_probs()

    def bprop(self):
        self.bprop_word_probs()
        self.dact = np.zeros(self.act.shape)
        self.dact = self.activation_function.derivative(self.h, self.dh)

        if self.hidden_bias_scaled_by_document_size:
            self.dc[:] = self.dact.sum(axis=0) * len(self.words)
        else:
            self.dc[:] = self.dact.sum(axis=0)
        dacc_input = np.zeros((len(self.words), self.hidden_size))
        np.add.accumulate(self.dact[:0:-1, :], axis=0, out=dacc_input[-2::-1, :])
        mllin.multiple_row_accumulate(dacc_input, self.words, self.dW)

    def update(self):
        self.update_word_probs()
        if self.normalize_by_document_size:
            self.c -= self.learning_rate * self.dc
            tm_utils.doc_mnade_sparse_update_W(self.W, self.word_ids, self.dW, self.learning_rate)
        else:
            self.c -= self.learning_rate * len(self.words) * self.dc
            tm_utils.doc_mnade_sparse_update_W(self.W, self.word_ids, self.dW, self.learning_rate * len(self.words))

    def fit(self, trainset):
        stage = 0
        while stage < self.n_stages:
            # TODO?
            #if stage == 0:
            #    self.initialize(trainset)
            stage += 1

            for word_counts, word_ids in trainset:
                self.fprop(word_ids, word_counts)
                self.bprop()
                self.update()

    def test(self, testset):
        costs = []
        outputs = self.use(testset)
        for output in outputs:
            costs += [[-np.mean(output)]]
        return outputs, costs

    def use(self, testset):
        rng_test_time = np.random.mtrand.RandomState(1234)
        tmp_rng = self.rng
        self.rng = rng_test_time

        outputs = []
        for word_counts, word_ids in testset:
            o = 0
            for i in range(self.testing_ensemble_size):
                self.fprop(word_ids, word_counts)
                o += np.log(self.probs).sum(axis=1)
            outputs += [o / self.testing_ensemble_size]
        self.rng = tmp_rng
        return outputs
