
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
parser.add_option('--checkpoint-path', dest='checkpoint_path', type=str, \
                  default='checkpoints/densenet-final.pth.tar')
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

def test(data_loader, model):
  correct = 0.0
  total = 0.0

  for data in data_loader:
    images, labels = data
    images_var = Variable(images, requires_grad=False, volatile=True)
    if torch.cuda.is_available:
      images_var = images_var.cuda()

    scores = model(images_var)
    _, predicted = torch.max(scores.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(correct)
    print(total)
    print()

  print('acc={}'.format(100 * correct / total))

def main():
  test_dataset = dsets.CIFAR10('dataset/', train=False, transform=preprocess_transform, download=True)
  
  state_dict = torch.load(open(options.checkpoint_path, 'rb'))
  model = DenseNetBC()
  model.load_state_dict(state_dict)

  if torch.cuda.is_available:
    model.cuda()

  test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)
  test(test_loader, model)

if __name__ == "__main__":
  main()
