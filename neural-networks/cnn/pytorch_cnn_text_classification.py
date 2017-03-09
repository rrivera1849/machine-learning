
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import numpy as np
import data_helpers
from tensorflow.contrib import learn

np.random.seed(10)

class TextCNN(nn.Module):
  def __init__ (self, vocab_size):
    super(TextCNN, self).__init__()
    self.embedding = nn.Embedding(vocab_size, 128)

    self.conv1 = nn.Conv2d(  1, 128, (3, 128, 1))
    self.conv2 = nn.Conv2d(128, 128, (4, 128, 1))
    self.conv3 = nn.Conv2d(128, 128, (5, 128, 1))

    self.fc1 = nn.Linear ((52*128) + (53*128) + (54*128), 2)

  def forward(self, x):
    embed = self.embedding(torch.LongTensor(x))
    pool1 = F.max_pool2d(F.relu(self.conv1(embed)), (1, 56))
    pool2 = F.max_pool2d(F.relu(self.conv2(embed)), (1, 56))
    pool3 = F.max_pool2d(F.relu(self.conv3(embed)), (1, 56))

    squashed_pool1 = self.pool1.view(-1, 54*1*128)
    squashed_pool2 = self.pool2.view(-1, 53*1*128)
    squashed_pool3 = self.pool3.view(-1, 52*1*128)

    squeezed = torch.cat((squashed_pool1, 
                          squashed_pool2, 
                          squashed_pool3), 1)

    x = self.fc1(squeezed)
    return x

current_running_loss = 0.0
current_train_step = 0

def train_step(model, X_batch, y_batch, criterion, optimizer):
  # Convert batches to torch variables so that we can feed
  # them in to our model. 
  X_batch, y_batch = Variable (X_batch), Variable(y_batch)
  optimizer.zero_grad()

  outputs = model(X_batch)
  loss = criterion(outputs, y_batch)

  loss.backward()
  optimizer.step()

  current_running_loss += loss.data[0]
  current_train_step += 1

  if current_train_step % 100 == 0:
    print '[%d] - %f' % (current_train_step, current_running_loss / 100.0)
    current_running_loss = 0.0

print 'Loading data'
x_text, y = data_helpers.load_data_and_labels ('./data/rt-polaritydata/rt-polarity.pos',
                                               './data/rt-polaritydata/rt-polarity.neg')

max_document_length = max ([len (x.split (" ")) for x in x_text])
print 'max_document_length =', max_document_length

vocab_processor = learn.preprocessing.VocabularyProcessor (max_document_length)
vocabulary_size = len(vocab_processor.vocabulary_)
x = np.array (list (vocab_processor.fit_transform (x_text)))

shuffle_indices = np.random.permutation (np.arange (len (y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int (0.3 * float (len (y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Vocabulary Size: {:d}".format (vocabulary_size))
print("Train/Dev split: {:d}/{:d}".format (len (y_train), len(y_dev)))

model = TextCNN(vocabulary_size)
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters())

batches = data_helpers.batch_iter (zip (x_train, y_train), 64, 10)

for batch in batches:
  x_batch, y_batch = zip (*batch)
  x_batch, y_batch = torch.Tensor(np.array(x_batch)), torch.Tensor(np.array(y_batch))
  train_step (model, x_batch, y_batch, criterion, optimizer)
