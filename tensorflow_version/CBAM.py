import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers, optimizers
import tensorflow.keras as keras


def regurlarized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding="same",
                         use_bias=False,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(5e-4))


class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()

        self.conv1 = layers.Conv2D(in_planes // ration, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)

        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True)

        def call(self, inputs):
            avg = self.avg(inputs)
            max = self.max(inputs)
            avg = layers.Reshape((1, 1, avg.shape[1]))(avg)
            max = layers.Reshape((1, 1, max.shape[1]))(max)
            avg_out = self.conv2(self.conv1(avg))
            max_out = self.conv2(self.conv1(max))
            out = avg_out + max_out
            out = tf.nn.sigmoid(out)
            return out


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regurlarized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)
        return out


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = regurlarized_padded_conv(out_channels, kernel_size=3,
                                              strides=stride)
        self.bn1 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.conv2 = regurlarized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # 3.判断stride是否等于1，如果为1就是没有降采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regurlarized_padded_conv(self.expansion * out_channels,
                                                                 kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])

        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=True):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 预测理卷积
        self.stem = Sequential([
            regurlarized_padded_conv(64, kernel_size=3, strides=1),
            layers.BatchNormalization()
        ])
        # 创建4个残差网络
        self.layer1 = self.build_resblock(32, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(64, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.final_bn = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=True):
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        out = self.final_bn(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out

    #         self.final_bn = layers.BatchNormalization()
    #         self.avgpool =
    # 1.创建resBlock
    def build_resblock(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks


def ResNet18():
    return ResNet([2, 2, 2, 2])


model = ResNet18()
model.build(input_shape=(None, 160, 160, 3))
model.summary()
