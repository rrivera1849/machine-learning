"""
The main training routine is located in this file.
First, we train the cumbersome model and then we distill the knowledge
to the smaller network and compare their performance. We expect the distilled
network to gain as much experience from the cumbersome or "mentor" model.
"""

import os
import datetime
from time import gmtime, strftime, time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

from model import Model

np.random.seed(int(time()))
DROPOUT_KEEP_PROB = 1

if not os.path.isdir('./data'):
  os.makedirs('./data')

timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print 'Saving output to {}'.format(out_dir)

checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
print 'Saving model checkpoints to {}'.format(checkpoint_dir)

if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)

print 'Fetching MNIST'
mnist = fetch_mldata('MNIST original', data_home='data/')

print 'Splitting Train/Test 80/20'
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=3)

y_train_one_hot = np.zeros((len(y_train), 10))
y_train_one_hot[np.arange(len(y_train)), np.int_(y_train)] = 1

y_test_one_hot = np.zeros((len(y_test), 10))
y_test_one_hot[np.arange(len(y_test)), np.int_(y_test)] = 1

print 'Length of training set {}'.format(len(X_train))
print 'Length of test set {}'.format(len(X_test))

def batch_iter(X, y, batch_size, num_epochs):
  data_size = len(X)
  num_batches_per_epoch = ((data_size - 1) / batch_size) + 1

  for epoch in range(num_epochs):
    shuffle_indices = np.random.permutation(np.arange(data_size))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    for batch in range(num_batches_per_epoch):
      start_index = batch * batch_size
      end_index = start_index + batch_size
      yield (X[start_index:end_index], y[start_index:end_index])

with tf.Graph().as_default():
  session_conf = tf.ConfigProto(device_count = {'GPU' : 0})
  session = tf.Session(config = session_conf)

  with session.as_default():
    global_step = tf.Variable(0, name="global_step", trainable=False)

    mentor_model = Model(600)

    learning_rate = tf.train.inverse_time_decay(0.001, global_step, 200, 0.5)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(mentor_model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def mentor_model_train_step(X_batch, y_batch):
      feed_dict = {
          mentor_model.input_x: X_batch,
          mentor_model.input_y: y_batch,
          mentor_model.dropout_keep_prob: DROPOUT_KEEP_PROB
          }

      _, step, loss, accuracy = session.run (
          [train_op, global_step, mentor_model.loss, mentor_model.accuracy],
          feed_dict
          )

      time_str = datetime.datetime.now().isoformat()
      print '{}: step {}, loss {:g}, accuracy {:g}'.format(time_str, step, loss, accuracy)

    def mentor_model_dev_step(X_batch, y_batch):
      feed_dict = {
          mentor_model.input_x: X_batch,
          mentor_model.input_y: y_batch,
          mentor_model.dropout_keep_prob: 1
          }

      step, loss, accuracy = session.run (
        [global_step, mentor_model.loss, mentor_model.accuracy],
        feed_dict
        )

      time_str = datetime.datetime.now().isoformat()
      print '{}: step {}, loss {:g}, accuracy {:g}'.format(time_str, step, loss, accuracy)

    saver = tf.train.Saver(tf.global_variables())
    session.run(tf.global_variables_initializer())

    batches = batch_iter(X_train, y_train_one_hot, 128, 10)
    for batch in batches:
      mentor_model_train_step(batch[0], batch[1])

      current_step = tf.train.global_step(session, global_step)
      if current_step % 100 == 0:
        print '\nMentor Evaluation\n'
        mentor_model_dev_step(X_test, y_test_one_hot)
        print '\n\n'
        path = saver.save(session, checkpoint_prefix, global_step = current_step)
