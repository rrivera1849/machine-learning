# This code was originally acquired from https://github.com/dennybritz/cnn-text-classification-tf
# This file has been modified by myself.
# Please contact rafaelriverasoto@gmail.com for more information.

import os
import data_helpers
import datetime
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from cnn_text_classification import TextCNN

# Data preprocessing parameters
tf.flags.DEFINE_float ("dev_sample_percentage", .1, "Percentage of training data to use for validation.")
tf.flags.DEFINE_string ("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string ("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer ("embedding_dim", 128, "Dimensionality of the character embedding (default: 128)")
tf.flags.DEFINE_string ("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer ("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float ("dropout_prob", 0.5, "Dropout probability (default: 0.5)")
tf.flags.DEFINE_float ("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0")

# Training parameters
tf.flags.DEFINE_integer ("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer ("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer ("evaluate_every", 100, "Evaluate the model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer ("checkpoint_every", 100, "Save the model after this many steps (default: 100)")
tf.flags.DEFINE_integer ("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean ("allow_soft_placement", True, 
                         "Allow device soft placement, this basically mean that code that was \
                          designed for a GPU will be allowed to run on a CPU and vice-versa \
                          without any errors.")
tf.flags.DEFINE_boolean ("log_device_placement", False, "Log placement of each operation on devices.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags ()

print ("\nParameters:")
for param, value in sorted (FLAGS.__flags.items ()):
  print ("{}={}".format (param.upper (), value))
print ("")

# ==================================================
# Data Preprocessing
# ==================================================

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels (FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary

# Calculate the maximum vocabulary length by splitting each word and
# counting the total. Choose the max.
max_document_length = max ([len (x.split (" ")) for x in x_text])

# VocabularyProcessor maps documents to sequences of word identifiers
vocab_processor = learn.preprocessing.VocabularyProcessor (max_document_length)

# Learns a vocabulary from our text and returns the indexes of words
x = np.array (list (vocab_processor.fit_transform (x_text)))

# Randomly shuffle data
np.random.seed(10)

# Generate a random permutation
shuffle_indices = np.random.permutation (np.arange (len (y)))

# Use the indexes of the random permutations to get our "shuffled" data
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set

# Get got a dev. sample percentage from the command line, use this to 
# determine the index where the dev set should start
dev_sample_index = -1 * int (FLAGS.dev_sample_percentage * float (len (y)))

x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Vocabulary Size: {:d}".format (len (vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format (len (y_train), len(y_dev)))

with tf.Graph ().as_default ():

  session_conf = tf.ConfigProto (
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
    )

  sess = tf.Session (config = session_conf)

  with sess.as_default ():
    # Initialize the TextCNN class
    cnn = TextCNN (
        sequence_length = x_train.shape[1],
        num_classes = y_train.shape[1],
        vocab_size = len (vocab_processor.vocabulary_),
        embedding_size = FLAGS.embedding_dim,
        filter_sizes = map (int, FLAGS.filter_sizes.split (",")),
        num_filters = FLAGS.num_filters
      )

    # We define how to optimize our loss function, we do so with the Adam optimizer
    
    # Global step that we're currently on
    global_step = tf.Variable (0, name = "global_step", trainable = False)

    optimizer = tf.train.AdamOptimizer (1e-3)
    grads_and_vars = optimizer.compute_gradients (cnn.loss)

    # Computes the gradient updates of our parameters
    train_op = optimizer.apply_gradients (grads_and_vars, global_step = global_step)

    # Summaries write the evolution of various quantities such as loss and accuracy
    # to disk using a SummaryWriter
    timestamp = str (int (time.time ()))
    out_dir = os.path.abspath (os.path.join (os.path.curdir, "runs", timestamp))
    print ("Writing to {}\n".format (out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar ("loss", cnn.loss)
    acc_summary = tf.summary.scalar ("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge ([loss_summary, acc_summary])
    train_summary_dir = os.path.join (out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter (train_summary_dir, sess.graph)

    # Dev Summaries
    dev_summary_op = tf.summary.merge ([loss_summary, acc_summary])
    dev_summary_dir = os.path.join ("out_dir", "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter (dev_summary_dir, sess.graph)

    # Checkpointing
    checkpoint_dir = os.path.abspath (os.path.join (out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join (checkpoint_dir, "model")
    # Tensorflow assumes that the checkpoint directory already exists, we should create it
    if not os.path.exists (checkpoint_dir):
      os.makedirs (checkpoint_dir)

    # The saver is used to checkpoint our model
    saver = tf.train.Saver (tf.global_variables ())

    # Initialize all variables before training
    sess.run (tf.global_variables_initializer ())

    def train_step (x_batch, y_batch):
      """ This function runs a single training step.

      We will run the computation graph and save our summaries as well as 
      print out some useful information.

      Keyword Arguments:
      x_batch, y_batch -- A batch of data & labels
      """
      feed_dict = {
          cnn.input_x : x_batch,
          cnn.input_y : y_batch,
          cnn.dropout_prob : FLAGS.dropout_prob
        }

      _, step, summaries, loss, accuracy = sess.run (
          [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
          feed_dict
        )

      time_str = datetime.datetime.now ().isoformat ()
      print ("{}: step {}, loss {:g}, acc {:g}".format (time_str, step, loss, accuracy))
      train_summary_writer.add_summary (summaries, step)

    def dev_step (x_batch, y_batch, writer = None):
      """Evaluates the model on an arbitrary set of data.

      We don't use training_op here and turn off dropout as well. This is 
      strictly for evaluation.
      """
      feed_dict = {
          cnn.input_x : x_batch,
          cnn.input_y : y_batch,
          cnn.dropout_prob : 1.0
        }

      step, summaries, loss, accuracy = sess.run (
          [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
          feed_dict
        )

      time_str = datetime.datetime.now ().isoformat ()
      print ("{}: step {}, loss {:g}, acc {:g}".format (time_str, step, loss, accuracy))
      if writer:
        writer.add_summary (summaries, step)

    # Training Loop -- Here we use the functions defined above to train & evaluate our model
    batches = data_helpers.batch_iter (
        zip (x_train, y_train),
        FLAGS.batch_size,
        FLAGS.num_epochs
      )

    for batch in batches:
      x_batch, y_batch = zip (*batch)
      train_step (x_batch, y_batch)

      current_step = tf.train.global_step (sess, global_step)
      if current_step % FLAGS.evaluate_every == 0:
        print ("\nEvaluation")
        dev_step (x_dev, y_dev, writer = dev_summary_writer)
        print ("")

      if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save (sess, checkpoint_prefix, global_step = current_step)
        print ("Saved model checkpoint to {}\n".format (path))
