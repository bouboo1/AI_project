import argparse
from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST

from AlexNet import AlexNet
from DenseNet import DenseNet
from LeNet import LeNet
from ResNet import ResNet
from VGGNet import VGGNet
from MobileNet import MobileNet

class Model:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='LeNet')
        parser.add_argument('--lr', default=0.006)
        parser.add_argument('--batch_size', default=256)
        parser.add_argument('--epoch', default=10)
        parser.add_argument('-dropout', type=float, default=0.5, help='dropout value')
        args = parser.parse_args()

        MODEL_= {'LeNet': LeNet(dropout = args.dropout), 'AlexNet': AlexNet(dropout = args.dropout), 'ResNet': ResNet(dropout = args.dropout), 
                      'VGGNet': VGGNet(dropout = args.dropout), 'DenseNet': DenseNet(dropout = args.dropout), 'MobileNet': MobileNet(dropout = args.dropout)}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.model = MODEL_[args.model].to(self.device)
        self.model_name = args.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(args.lr))
        self.batch_size = int(args.batch_size)
        self.epoch = int(args.epoch)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.train_loss_record = []
        self.val_loss_record = []
        self.val_acc_record = []

    def load_dataset(self):
        train_data = MNIST('./data/mnist',
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
                           ]))
        test_data = MNIST('./data/mnist',
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
                          ]))
        train_size = int(0.8 * len(train_data))
        test_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, test_size])

        self.train_set = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=10)
        self.val_set = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=10)
        self.test_set = DataLoader(test_data, batch_size=self.batch_size, num_workers=10)

    def train(self, epoch):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        loss_epoch = 0
        i = 0

        for i, (images, labels) in enumerate(self.train_set):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            if i % 10 == 0:
                print('Train - Epoch: %d, Batch: %d, Loss: %f' % (epoch, i, loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            loss_epoch += loss.cpu().detach().numpy()
        loss_epoch /= i

        self.model.eval()
        all_correct = 0
        all_sample = 0
        for i, (images, labels) in enumerate(self.val_set):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.model(images.float())
            self.val_loss_record.append(criterion(pred, labels).cpu().detach().numpy())
            current_correct = torch.eq(torch.argmax(F.softmax(pred, dim=1), dim=1), labels)
            all_correct += current_correct.sum()
            all_sample += current_correct.shape[0]
            self.val_acc_record.append((all_correct / all_sample).cpu().detach().numpy())
        print('Validation - Epoch: %d, Accuracy: %f' % (epoch + 1, all_correct / all_sample))

    def test(self):
        self.model.eval()
        all_correct = 0
        all_sample = 0
        for i, (images, labels) in enumerate(self.test_set):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = torch.argmax(F.softmax(self.model(images.float()), dim=1), dim=1)
            current_correct = torch.eq(pred, labels)
            all_correct += current_correct.sum()
            all_sample += current_correct.shape[0]
        print('Test - Accuracy: %f' % (all_correct / all_sample))

    def run(self):
        for epoch in range(self.epoch):
            self.train(epoch)
        self.test()
        summary(self.model, (64, 1, 28, 28), device='cpu')

    def save(self):
        torch.save(self.model, self.model_name)

if __name__ == "__main__":
    RunModel = Model()
    RunModel.load_dataset()
    RunModel.run()
    RunModel.save()