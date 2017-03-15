
import numpy as np
import tensorflow as tf

class SimpleChainCNN(object):
  def __init__(self, num_classes):
    X = tf.placeholder(tf.float32, [None, 128, 128, 1], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='y')
    print 'X', X.get_shape()
    print 'y', y.get_shape()

    last_conv = self._build_chain_conv(X, 3, 32, 3, False)
    last_conv = self._build_chain_conv(last_conv, 4, 64, 3, True)
    last_conv = self._build_chain_conv(last_conv, 5, 128, 3, True)

    max_pool = tf.nn.avg_pool (last_conv, ksize=[1, 101,101,128], strides=[1,1,1,1], padding='VALID', name='max-pool')
    squashed = tf.reshape (max_pool, shape=[-1, 128])
    print 'Max Pool', max_pool.get_shape()
    print 'Squashed', squashed.get_shape()

    with tf.name_scope ('local') as scope:
      W = tf.Variable (tf.truncated_normal ([128,num_classes], stddev=0.1), name='W')
      b = tf.Variable (tf.constant (0.1, shape=[num_classes]), name='b')
      self.scores = tf.add(tf.matmul(squashed, W), b, name='scores')
      self.predictions = tf.argmax (self.scores, 1, name='predictions')

    print 'Scores', self.scores.get_shape()
    print 'Predictions', self.predictions.get_shape()

    with tf.name_scope('loss') as scope:
      losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, y)
      self.loss = tf.reduce_mean(losses, name='loss')

    with tf.name_scope('accuracy') as scope:
      correct_predictions = tf.equal(self.predictions, tf.argmax(y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


  def _build_chain_conv(self, start_conv, filter_dim, num_filters, chain_length, cont=False):
    for chain in range(chain_length):
      with tf.name_scope('conv-%d-%d' % (filter_dim, chain+1)) as scope:
        if not cont and chain == 0:
          kernel = tf.Variable(
                    tf.truncated_normal(
                      [filter_dim, filter_dim, 1, num_filters], 
                      stddev=0.1), 
                    name='kernel-%d-%d' % (filter_dim,chain+1))

          conv = tf.nn.conv2d(start_conv, kernel, 
                              strides=[1,1,1,1], padding='VALID', 
                              name='conv-%d-%d' % (filter_dim,chain+1))
        elif cont and chain == 0:
          kernel = tf.Variable(
                    tf.truncated_normal(
                      [filter_dim, filter_dim, num_filters / 2, 2], 
                      stddev=0.1), 
                    name='kernel-%d-%d' % (filter_dim,chain+1))

          conv = tf.nn.depthwise_conv2d(start_conv, kernel, 
                                        strides=[1,1,1,1], padding='VALID', 
                                        name='conv-%d-%d' % (filter_dim,chain+1))
        else:
          kernel = tf.Variable(
                    tf.truncated_normal(
                      [filter_dim, filter_dim, num_filters, 1], 
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


SimpleChainCNN(4);
