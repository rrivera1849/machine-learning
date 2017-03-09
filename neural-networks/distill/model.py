"""
This file handles the creation of the models specified in the paper named
"Distilling the Knowledge of a Neural Network". Specifically, we replicate
the architecture specified in Section 3.
"""
import tensorflow as tf

class Model(object):
  def __init__(self, hlength):
    """Creates a model with two hidden layers as described in the paper.

    Architecture:
      Two hidden layers with a total of 2*hlength ReLU activation units thus
      hlength neurons per layer.

      [INPUT_X] -> [784,hlength] -> [hlength,hlength] -> [hlength,10] -> [CLASSIFICATION]

    Keyword Arguments:
      hlength: number of neurons to allocate in hidden layer
    """
    self.input_x = tf.placeholder(tf.float32, shape = [None, 784], name = "input_x")
    self.input_y = tf.placeholder(tf.int32, shape = [None, 10], name = "input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    with tf.name_scope("local_1") as scope:
      W = tf.Variable(tf.truncated_normal(shape = [784,hlength], stddev=0.1), name = "W")
      b = tf.Variable(tf.constant(0.1, shape=[hlength]), name = "b")
      pre_activation = tf.add(tf.matmul(self.input_x, W), b, name="pre_activation")
      self.local1 = tf.nn.relu(pre_activation, name="local1")

    with tf.name_scope("dropout_1") as scope:
      self.local1 = tf.nn.dropout(self.local1, self.dropout_keep_prob)

    with tf.name_scope("local_2") as scope:
      W = tf.Variable(tf.truncated_normal(shape = [hlength, hlength], stddev=0.1), name = "W")
      b = tf.Variable(tf.constant(0.1, shape=[hlength]), name="b")
      pre_activation = tf.add(tf.matmul(self.local1, W), b, name="pre_activation")
      self.local2 = tf.nn.relu(pre_activation)

    with tf.name_scope("dropout_2") as scope:
      self.local2 = tf.nn.dropout(self.local2, self.dropout_keep_prob)

    with tf.name_scope("output") as scope:
      W = tf.Variable(tf.truncated_normal(shape = [hlength, 10], stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape = [10]), name = "b")
      self.output = tf.add(tf.matmul(self.local2, W), b, name="output")
      self.predictions = tf.argmax(self.output, axis=1, name="predictions")

    with tf.name_scope("loss") as scope:
      losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.input_y)
      self.loss = tf.reduce_mean(losses, name="loss")

    with tf.name_scope("accuracy") as scope:
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

