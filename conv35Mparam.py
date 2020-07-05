import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
from dataset_loader import MyDataset
from torch.utils.data import DataLoader
from torchsummary import summary


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512 * 8 * 8, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 8 * 8)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

    def train_sgd(self, device):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        loss = nn.CrossEntropyLoss()
        initepoch = 0
        min_loss = 90

        print("\n")
        print("starting training!!...")
        print("\n")
        for epoch in range(initepoch, 80):  # loop over the dataset multiple times
            # timestart = time.time()

            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                # print statistics
                running_loss += l.item()
                # print("i ",i)
                if i % 8 == 7:  # print every 500 mini-batches
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Epoch: %d .Mini-batches: %d .Training loss: %.4f. Accuracy of the network on the %d '
                          'train images: %.3f %%\n' % (epoch, i, running_loss / 8, total,
                                                       100.0 * correct / total))
                    total = 0
                    correct = 0

            min_loss = self.test(device, min_loss, epoch, loss)
        # print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))
        print("\n")
        print('Finished Training\n')

    @torch.no_grad()
    def test(self, device, min_loss, epoch, loss):
        correct = 0
        total = 0
        val_loss = 0
        i=0
        with torch.no_grad():
            self.eval()
            for data in testloader:
                i = i + 1
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                l = loss(outputs, labels)
                val_loss += l.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss/i
        acc = 100.0 * correct / total
        if acc > min_loss:
            min_loss = acc
            print("save model\n")
            torch.save(net.state_dict(), 'model_main.pth')
        print('epoch: %d - Loss on the test set is %.4f. Accuracy of the network on the %d test images: %.3f %%\n' % (
            epoch, val_loss, total,
            acc))
        return min_loss

    @torch.no_grad()
    def eval1(self, device):
        correct = 0
        total = 0
        with torch.no_grad():
            self.eval()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %.3f %%' % (total,
                                                                          100.0 * correct / total))


train_dataset_file = 'train_data.txt'
test_dataset_file = 'test_data.txt'
noise_data_file = '0dbfan2_data.txt'
#
transform = transforms.Compose([  # transforms.ToPILImage(),             # 将ndarray转化成 pillow的Image格式
    transforms.Resize((160, 160)),  # 裁减至（256,512）
    transforms.ToTensor()])  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]，而且会将[
# w,h,c]转化成pytorch需要的[c,w,h]格式
train_dataset = MyDataset(train_dataset_file, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
# test_dataset = MyDataset(test_dataset_file, transform=transform)
# # print(test_dataset.__len__())
# testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_dataset = MyDataset(noise_data_file, transform=transform)
print(test_dataset.__len__())
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
summary(net, (3, 160, 160), device='cpu')
net = net.to(device)
# net.train_sgd(device)
net.load_state_dict(torch.load('model_main.pth'))
# net.eval()
net.eval1(device)

