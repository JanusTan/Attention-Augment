from collections import OrderedDict
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

from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.models.densenet import _DenseLayer, _DenseBlock

import numpy as np

from torchsummary import summary


# --------------------
# Attention augmented convolution
# --------------------

class AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dk, dv, nh, relative, input_dims, **kwargs):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.nh = nh
        self.relative = relative

        assert dk % nh == 0, 'nh must divide dk'
        assert dv % nh == 0, 'nh must divide dv'

        # `same` conv since conv and attn are concatenated in the output
        padding = kwargs.pop('padding', None)
        if not padding: padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride, padding, bias=False,
                              **kwargs) if out_channels > dv else None
        self.in_proj_qkv = nn.Conv2d(in_channels, 2 * dk + dv, kernel_size=1, stride=stride, bias=False)
        self.out_proj = nn.Conv2d(dv, dv, kernel_size=1, bias=False)

        if relative:
            H, W = input_dims
            self.key_rel_h = nn.Parameter(dk ** -0.5 + torch.randn(dk // nh, 2 * H - 1))
            self.key_rel_w = nn.Parameter(dk ** -0.5 + torch.randn(dk // nh, 2 * W - 1))

    def rel_to_abs(self, x):
        B, nh, L, _ = x.shape  # (B, nh, L, 2*L-1)

        # pad to shift from relative to absolute indexing
        x = F.pad(x, (0, 1))  # (B, nh, L, 2*L)
        x = x.flatten(2)  # (B, nh, L*2*L)
        x = F.pad(x, (0, L - 1))  # (B, nh, L*2*L + L-1)

        # reshape and slice out the padded elements
        x = x.reshape(B, nh, L + 1, 2 * L - 1)
        return x[:, :, :L, L - 1:]

    def relative_logits_1d(self, q, rel_k):
        B, nh, H, W, dkh = q.shape

        rel_logits = torch.matmul(q, rel_k)  # (B, nh, H, W, 2*W-1)
        # collapse height and heads
        rel_logits = rel_logits.reshape(B, nh * H, W, 2 * W - 1)
        rel_logits = self.rel_to_abs(rel_logits)  # (B, nh*H, W, W)
        # shape back and tile height times
        return rel_logits.reshape(B, nh, H, 1, W, W).expand(-1, -1, -1, H, -1, -1)  # (B, nh, H, H, W, W)

    def forward(self, x):
        # compute qkv
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.split([self.dk, self.dk, self.dv], dim=1)
        # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
        B, _, H, W = qkv.shape
        flat_q = q.reshape(B, self.nh, self.dk // self.nh, H, W).flatten(3) * (self.dk // self.nh) ** -0.5
        flat_k = k.reshape(B, self.nh, self.dk // self.nh, H, W).flatten(3)
        flat_v = v.reshape(B, self.nh, self.dv // self.nh, H, W).flatten(3)

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # (B, nh, HW, HW)
        if self.relative:
            q = flat_q.reshape(B, self.nh, self.dk // self.nh, H, W).permute(0, 1, 3, 4, 2)  # (B, nh, H, W, dkh)
            # compute relative logits in width dim
            w_rel_logits = self.relative_logits_1d(q, self.key_rel_w)  # (B, nh, H, H, W, W)
            # repeat for heigh dim by transposing H,W and then permuting output
            h_rel_logits = self.relative_logits_1d(q.transpose(2, 3), self.key_rel_h)  # (B, nh, W, W, H, H)
            # permute and reshape for adding to the attention logits
            w_rel_logits = w_rel_logits.permute(0, 1, 2, 4, 3, 5).reshape(B, self.nh, H * W, H * W)
            h_rel_logits = h_rel_logits.permute(0, 1, 4, 2, 5, 3).reshape(B, self.nh, H * W, H * W)
            # add to attention logits
            logits += h_rel_logits + w_rel_logits

        self.weights = F.softmax(logits, -1)

        attn_out = torch.matmul(self.weights, flat_v.transpose(2, 3))  # (B, nh, HW, dvh)
        attn_out = attn_out.transpose(2, 3)  # (B, nh, dvh, HW)
        attn_out = attn_out.reshape(B, -1, H, W)  # (B, dv, H, W)
        attn_out = self.out_proj(attn_out)

        if self.conv is not None:
            return torch.cat([self.conv(x), attn_out], dim=1)
        else:
            return attn_out

    def extra_repr(self):
        return 'dk={dk}, dv={dv}, nh={nh}, relative={relative}'.format(**self.__dict__)


# --------------------
# Attention Augmented ResNet
# --------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, input_dims=None, attn_params=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # attention
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20 * nh, int((attn_params['k'] * width // nh) * nh))
            dv = int((attn_params['v'] * width // nh) * nh)
            relative = attn_params['relative']
            # scale input dims to network HW outputs at this layer
            input_dims = int(attn_params['input_dims'][0] * 16 / planes), int(
                attn_params['input_dims'][1] * 16 / planes)
            # print('Bottleneck attention: dk {}, dv {}, input_dims {}x{}'.format(dk, dv, *input_dims))

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) if attn_params is None else \
            AAConv2d(width, width, 3, stride, dk, dv, nh, relative, input_dims, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ cf paper -- replaces the first conv3x3 in BasicBlock and Bottleneck of Resnet layers 2,3,4;
    ResNet class from torchvision. """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attn_params=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       attn_params=attn_params)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       attn_params=attn_params)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       attn_params=attn_params)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attn_params=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attn_params=attn_params))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attn_params=attn_params))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
        i = 0
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
        val_loss = val_loss / i
        acc = 100.0 * correct / total
        if acc > min_loss:
            min_loss = acc
            print("save model\n")
            torch.save(net.state_dict(), 'model_aares121.pth')
        print(
            'epoch: %d - Loss on the test set is %.4f. Accuracy of the network on the %d test images: %.3f %%\n' % (
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
net = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=3,
             attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (160, 160)})
# resnet50 layers [3,4,6,3]; resnet101 layers [3,4,23,3]; resnet 152 layers [3,8,36,3]
summary(net, (3, 160, 160), device='cpu')
net = net.to(device)
# net.train_sgd(device)
net.load_state_dict(torch.load('model_aares121.pth'))
# net.eval()
net.eval1(device)
