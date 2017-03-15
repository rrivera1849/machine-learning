
import numpy as np
import tensorflow as tf

class SimpleChainCNN(object):
  def __init__(self, num_classes):
    X = tf.placeholder(tf.float32, [None, 128, 128, 1], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='y')
    print 'X', X.get_shape()
    print 'y', y.get_shape()


    with tf.name_scope('conv3-0') as scope:
      kernel = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1), name='kernel-3-0')
      conv = tf.nn.conv2d(X, kernel, strides=[1,1,1,1], padding='VALID', name='conv-3-0')
      biases = tf.Variable(tf.constant(0.1, shape=[16]), name='bias-3-0')
      pre_activation = tf.nn.bias_add(conv, biases)
      activation = tf.nn.relu(pre_activation, name='activation-3-0')

      print 'Conv-3-0', conv.get_shape()
      print 'Activation-3-0', activation.get_shape()
      self.last_chain_1 = self._build_chain_conv(activation, 3, 16, 2)

    with tf.name_scope('4-0') as scope:
      kernel = tf.Variable(tf.truncated_normal([4,4,16,2], stddev=0.1), name='kernel-4-0')
      conv = tf.nn.depthwise_conv2d(self.last_chain_1, kernel, strides=[1,1,1,1], padding='VALID', name='conv-4-0')
      biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases-4-0')
      pre_activation = tf.nn.bias_add(conv, biases)
      activation = tf.nn.relu(pre_activation, name='activation-4-0')

      print 'Conv-4-0', conv.get_shape()
      print 'Activation-4-0', activation.get_shape()
      self.last_chain_2 = self._build_chain_conv(activation, 4, 32, 2)

    max_pool = tf.nn.avg_pool (self.last_chain_2, ksize=[1, 113,113,32], strides=[1,1,1,1], padding='VALID', name='max-pool')
    squashed = tf.reshape (max_pool, shape=[-1, 32])
    print 'Max Pool', max_pool.get_shape()
    print 'Squashed', squashed.get_shape()

    with tf.name_scope ('local') as scope:
      W = tf.Variable (tf.truncated_normal ([32,num_classes], stddev=0.1), name='W')
      b = tf.Variable (tf.constant (0.1, shape=[num_classes]), name='b')

      # Wx + b = score
      self.scores = tf.add(tf.matmul(squashed, W), b, name='scores')
      print 'Scores', self.scores.get_shape()

      # Prediction = argmax (score)
      self.predictions = tf.argmax (self.scores, 1, name='predictions')
      print 'Predictions', self.predictions.get_shape()

    with tf.name_scope('loss') as scope:
      losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, y)
      self.loss = tf.reduce_mean(losses, name='loss')

    with tf.name_scope('accuracy') as scope:
      correct_predictions = tf.equal(self.predictions, tf.argmax(y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


  def _build_chain_conv(self, start_conv, filter_dim, num_filters, chain_length):
    for chain in range(chain_length):
      with tf.name_scope('conv-%d-%d' % (filter_dim, chain+1)) as scope:
        kernel = tf.Variable(
                   tf.truncated_normal(
                     [filter_dim,filter_dim,num_filters, 1], 
                     stddev=0.1), 
                   name='kernel-%d-%d' % (filter_dim,chain+1))

        conv = tf.nn.depthwise_conv2d(start_conv, kernel, 
                                      strides=[1,1,1,1], padding='VALID', 
                                      name='conv-%d-%d' % (filter_dim,chain+1))

        biases = tf.Variable(tf.constant(0.1, shape=[num_filters]), 
                             name='bias-%d-%d' % (filter_dim,chain+1))

        pre_activation = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(pre_activation, 
                                name='activation-%d-%d' % (filter_dim,chain+1))

        print 'Conv-%d-%d' % (filter_dim,chain+1), conv.get_shape()
        print 'Activation', activation.get_shape()
        start_conv = activation

    return start_conv


SimpleChainCNN(2);
