# Based in https://github.com/tensorflow/ranking/blob/b28653e110ad2bf780fb0cf083718f4a337a57bb/tensorflow_ranking/python/metrics.py#L231-L258

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def is_label_valid(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = ops.convert_to_tensor(labels)
  return math_ops.greater_equal(labels, 0)


def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
  """Prepares and validates the parameters.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.

  Returns:
    (labels, predictions, weights, topn) ready to be used for metric
    calculation.
  """
  labels = ops.convert_to_tensor(labels)
  predictions = ops.convert_to_tensor(predictions)
  weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
  example_weights = array_ops.ones_like(labels, dtype=tf.float32) * weights
  predictions.get_shape().assert_is_compatible_with(example_weights.get_shape())
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  predictions.get_shape().assert_has_rank(2)
  if topn is None:
    topn = array_ops.shape(predictions)[1]

  # All labels should be >= 0. Invalid entries are reset.
  is_label_valid_ = is_label_valid(labels)
  labels = array_ops.where(is_label_valid_, labels, array_ops.zeros_like(labels))
  predictions = array_ops.where(
      is_label_valid_, predictions,
      -1e-6 * array_ops.ones_like(predictions) + math_ops.reduce_min(
          predictions, axis=1, keepdims=True))
  return labels, predictions, example_weights, topn


def mean_reciprocal_rank(labels, predictions, weights=None, name=None):
  """Computes mean reciprocal rank (MRR).

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted mean reciprocal rank of the batch.
  """
  with ops.name_scope(name, 'mean_reciprocal_rank', (labels, predictions, weights)):
    _, list_size = array_ops.unstack(array_ops.shape(predictions))
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, list_size)
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn)
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = math_ops.to_float(math_ops.greater_equal(sorted_labels, 1))
    reciprocal_rank = 1.0 / math_ops.to_float(math_ops.range(1, topn + 1))
    # MRR has a shape of [batch_size, 1]
    mrr = math_ops.reduce_max(
        relevance * reciprocal_rank, axis=1, keepdims=True)
    return tf.reduce_mean(mrr)
    #return mrr, array_ops.ones_like(weights), mrr * array_ops.ones_like(weights)
    #return metrics.mean(mrr * array_ops.ones_like(weights), weights)




def sort_by_scores(scores, features_list, topn=None):
  """Sorts example features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s with the same shape as scores to be
      sorted.
    topn: An integer as the cutoff of examples in the sorted list.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  scores = ops.convert_to_tensor(scores)
  scores.get_shape().assert_has_rank(2)
  batch_size, list_size = array_ops.unstack(array_ops.shape(scores))
  if topn is None:
    topn = list_size
  topn = math_ops.minimum(topn, list_size)
  _, indices = nn_ops.top_k(scores, topn, sorted=True)
  list_offsets = array_ops.expand_dims(
      math_ops.range(batch_size) * list_size, 1)
  # The shape of `indices` is [batch_size, topn] and the shape of
  # `list_offsets` is [batch_size, 1]. Broadcasting is used here.
  gather_indices = array_ops.reshape(indices + list_offsets, [-1])
  output_shape = array_ops.stack([batch_size, topn])
  # Each feature is first flattened to a 1-D vector and then gathered by the
  # indices from sorted scores and then re-shaped.
  return [
      array_ops.reshape(
          array_ops.gather(array_ops.reshape(feature, [-1]), gather_indices),
          output_shape) for feature in features_list
  ]




if __name__ == '__main__':
    tf.enable_eager_execution()

    labels = tf.convert_to_tensor([
        [0, 0, 1],
        [1, 0, 0],
    ])
    values = tf.convert_to_tensor([
        [0.1, .2, .3],
        [0.1, .2, .4],
    ])

    print('Expected:', (1/1 + 1/3)/2)
    print('Obtained:', mean_reciprocal_rank(labels, values))
