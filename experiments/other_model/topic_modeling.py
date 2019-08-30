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


"""
The ``learners.topic_modeling`` module contains Learners meant for topic modeling
problems. The MLProblems for these Learners should be iterators over inputs,
which correspond to sparse representations of words counts:
they are pairs ``(freq,words)`` where ``freq`` is the vector
of frequences for each word present in the document, and ``words`` is the vector
of IDs for those words.

The currently implemented algorithms are:

* TopicModel:             the interface for all TopicModel objects that learn representation for documents.
* ReplicatedSoftmax:      the Replicated Softmax undirected topic model.
* DocNADE:                the Document Neural Autoregressive Distribution Estimator (DocNADE).
* InformationRetrieval:   applieds a given TopicModel to an information retrieval task.
"""


class Learner:
    """
    Root class or interface for a learning algorithm.

    All Learner objects inherit from this class. It is meant to
    standardize the behavior of all learning algorithms.

    """

    # def __init__():

    def train(self, trainset):
        """
        Runs the learning algorithm on ``trainset``
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def forget(self):
        """
        Resets the Learner to its original state.
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def use(self, dataset):
        """
        Computes and returns the output of the Learner for
        ``dataset``. The method should return an iterator over these
        outputs.
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def test(self, dataset):
        """
        Computes and returns the outputs of the Learner as well as the cost of
        those outputs for ``dataset``. The method should return a pair of two iterators, the first
        being over the outputs and the second over the costs.
        """
        raise NotImplementedError("Subclass should have implemented this method.")


class TopicModel(Learner):
    """
    Interface for all TopicModel objects that learn representation for documents.

    The only additional requirement from Learner is to define a method
    ``compute_document_representation(example)`` that outputs the representation
    for some given example (a ``(freq,words)`` pair).
    """

    def compute_document_representation(self, example):
        """
        Return the document representation of some given example.
        """

        raise NotImplementedError("Subclass should have implemented this method")
