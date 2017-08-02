
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from model import DenseNetBC

from optparse import OptionParser
parser = OptionParser()
parser.add_option('--data-augment', action='store_true', dest='data_augment', default=False, \
                  help='Whether or not to use data augmentation transforms')
(options, args) = parser.parse_args()

if options.data_augment:
  preprocess_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean = [ x / 255.0 for x in [125.3, 123.0, 113.9] ],
                           std = [ x / 255.0 for x in [63.0, 62.1, 66.7] ])
    ])
else:
  preprocess_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean = [ x / 255.0 for x in [125.3, 123.0, 113.9] ],
                           std = [ x / 255.0 for x in [63.0, 62.1, 66.7] ])
    ])

def train_step(images, labels, model, optimizer, criterion):
  optimizer.zero_grad()

  images_var = Variable(images, requires_grad=True, volatile=False)
  labels_var = Variable(labels, requires_grad=False, volatile=False)

  if torch.cuda.is_available:
    images_var = images_var.cuda()
    labels_var = labels_var.cuda()

  scores = model(images_var)

  loss = criterion(scores, labels_var)
  loss.backward()
  optimizer.step()

  return loss.data[0]

def adjust_lr(optimizer):
  for param_group in optimizer.param_groups:
    param_group['lr'] /= 10

def train(data_loader, model, optimizer, criterion):
  print_every = 100
  print_every_loss = 0.0

  iteration = 0
  for epoch in range(300):
    if int(epoch+1) == 150 or int(epoch+1) == 225:
      adjust_lr(optimizer)

    for images, labels in data_loader:
      loss = train_step(images, labels, model, optimizer, criterion)
      print_every_loss += loss
      iteration += 1

      if iteration % print_every == 0:
        print('Loss Iteration {} = {}'.format(iteration, print_every_loss / print_every), flush=True)
        print_every_loss = 0.0

def main():
  train_dataset = dsets.CIFAR10('dataset/', train=True, transform=preprocess_transform, download=True)

  model = DenseNetBC()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
  criterion = nn.CrossEntropyLoss()

  if torch.cuda.is_available:
    model.cuda()

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
  train(train_loader, model, optimizer, criterion)

  torch.save(model.state_dict(), open('checkpoints/densenet-final.pth.tar', 'wb'))

if __name__ == "__main__":
  main()
