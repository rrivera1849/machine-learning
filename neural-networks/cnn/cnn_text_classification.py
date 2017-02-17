# This code was originally acquired from https://github.com/dennybritz/cnn-text-classification-tf
# This file has been modified by myself.
# Please contact rafaelriverasoto@gmail.com for more information.

import tensorflow as tf
import numpy as np

class TextCNN (object):
    """CNN used for text classification.
    
    Architecture:
        [EMBEDDING] -> [CONVOLUTIONAL] -> [MAX_POOL] -> [SOFTMAX]
    """
    
    def __init__ (self, sequence_length, num_classes,                   vocab_size, embedding_size, filter_sizes, num_filters):
        """Initializes the model.
        
        Keyword Arguments:
        sequence_length - length of our sentences
        num_classes - number of classes (2 for us)
        vocab_size - total size of the vocabulary
        filter_sizes - number of words we want the convolutions to cover
                       should be an array: [1,2,3]
        num_filters - number of filters *per* filter size
        """
        
        # Placeholders for X & Y
        #   Dimensionality:
        #      X = [batch_size, sequence_length]
        #      Y = [batch_size, num_classes]
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        
        # Probability that a neuron is active or not
        self.dropout_prob = tf.placeholder (tf.float32, name="dropout_prob")
        
        # Forces execution on the CPU & creates a name scope for easy viewing
        with tf.device ("/cpu:0"), tf.name_scope ("embedding"):
            # W -- Embedding Matrix
            #   Dimensions:
            #     [vocab_size, embedding_size]
            W = tf.Variable (
                tf.random_uniform ([vocab_size, embedding_size], -1.0, 1.0),
                name = "W"
            )
            
            # Looks up our input inside the embedding matrix
            # Returns Dimensions: [batch_size, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup (W, self.input_x)
            
            # Expands our embedding lookup vector to include channel
            # Returns Dimensions: [batch_size, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims (self.embedded_chars, -1)
            
        # Output after applying max-pooling
        pooled_outputs = []
        
        # In this loop we're performing our convolution operations followed by
        # max-pooling. 
        #
        # Keep in mind that pooling gives you invariance to translation, 
        # rotation and scaling.
        #
        # The above applies to CNN's in general, because we're sliding our 
        # convolution accross the whole matrix we don't care where the 
        # object of interest is.
        #
        # One other advantage to the convolution operation is that it composes
        # higher-level features such as edges & shapes from lower level features
        # such as pixels.
        for index, filter_size in enumerate (filter_sizes):
            with tf.name_scope ("conv-maxpool-%s" % filter_size):
                # Take an example where filter_size = 2 and num_filters = 3
                #
                # Our "stacked" filter shape must be the size of the filter
                # followed by the shape of what we're applying the filter to.
                #
                # In this case we apply the filter to an embedding of dimension
                # [embedding_size, 1]. We "stack" our num_filters together with
                # the last dimension.
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                
                # W is our initial filter which is initialized to a truncated
                # normal distribution with std dev of 0.1
                W = tf.Variable (tf.truncated_normal (filter_shape, stddev=0.1), name="W")
                
                # We have a bias for each filter (num_filters)
                b = tf.Variable (tf.constant (0.1, shape=[num_filters], name = "b"))
                
                # Here we apply the convolution operation with strides of 1 and a
                # narrow convolution.
                conv = tf.nn.conv2d (
                    self.embedded_chars_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv"
                )
                
                # Apply some non-linearity to our convolution
                h = tf.nn.relu (tf.nn.bias_add (conv, b), name="relu")
                
                pooled = tf.nn.max_pool (
                    h,
                    ksize = [1, sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "pool"
                )
                
                pooled_outputs.append (pooled)
        
        # Flatten all the pooled outputs to [batch_size, num_filters_total]
        num_filters_total = num_filters * len (filter_sizes)
        self.h_pool = tf.concat (3, pooled_outputs)
        self.h_pool_flat = tf.reshape (self.h_pool, [-1, num_filters_total])
        
        # Apply a drop-out layer. This mechanism simply chooses which neurons
        # to keep active during training. It helps us avoid overfitting.
        with tf.name_scope ("dropout"):
            self.h_drop = tf.nn.dropout (self.h_pool_flat, self.dropout_prob)
        
        # Simply perform matrix multiplication to derive scores
        with tf.name_scope ("output"):
            W = tf.Variable (tf.truncated_normal ([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable (tf.constant (0.1, shape=[num_classes]), name="b")
            
            # Wx + b = score
            self.scores = tf.nn.xw_plus_b (self.h_drop, W, b, name="scores")
            # Prediction = argmax (score)
            self.predictions = tf.argmax (self.scores, 1, name="predictions")
            
        # We will use the cross_entropy_loss to define how good we're doing
        with tf.name_scope ("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits (self.scores, self.input_y)
            self.loss = tf.reduce_mean (losses)
            
        # Keep track of the accuracy of our network
        with tf.name_scope ("accuracy"):
            correct_predictions = tf.equal (self.predictions, tf.argmax (self.input_y, 1))
            self.accuracy = tf.reduce_mean (tf.cast (correct_predictions, "float"), name = "accuracy")
