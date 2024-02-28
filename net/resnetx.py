
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# from net.trained_resnet import trained_model

class ResidualBlock(nn.Module):
    """ Resnet Block"""

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in: in channel
        :param ch_out: out channel
        :param stride: c增大，h, w减小, p不变
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),  # 调整步幅以匹配out的形状
            nn.BatchNorm2d(ch_out)
        )

        if ch_out != ch_in:
            # [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x:[b, ch, h ,w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # element-size add: [b, ch_in, h, w] + [b, ch_out, h, w]
        out = self.extra(x) + out
        out = F.relu(out)

        return out

class CrossResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(CrossResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )
        self.extra3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )
        self.conv3 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(ch_out)
        self.conv4 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(ch_out)
        self.extra2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )
        self.extra4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )
    def forward(self, x1, x2):
        """

        :param x:[b, ch, h ,w]
        :return:
        """
        out1 = F.relu(self.bn1(self.conv1(x1)))
        out1 = self.bn2(self.conv2(out1))
        # short cut
        # element-size add: [b, ch_in, h, w] + [b, ch_out, h, w]
        out1 = self.extra1(x1) + out1 + self.extra3(x2)
        out1 = F.relu(out1)
        out2 = F.relu(self.bn3(self.conv1(x2)))
        out2 = self.bn4(self.conv2(out2))
        # short cut
        # element-size add: [b, ch_in, h, w] + [b, ch_out, h, w]
        out2 = self.extra2(x2) + out2 + self.extra4(x1)
        out2 = F.relu(out2)

        return out1, out2


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # pretreatment layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # follow 4 blocks
        self.blk1 = ResidualBlock(32, 64, stride=2)
        self.blk2 = ResidualBlock(64, 128, stride=2)
        self.blk3 = ResidualBlock(128, 256, stride=2)
        self.blk4 = ResidualBlock(256, 512, stride=2)

        # [b, 512, 1, 1]
        self.fc = nn.Linear(512 * 1 * 1, 5)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        print(x.size())
        x = F.adaptive_avg_pool2d(x, [1, 1])
        print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def test():
#     blk = ResidualBlock(64, 128)
#     tmp = torch.randn(2, 64, 224, 224)
#     out = blk(tmp)
#     print("block: ", out.shape)
#
#     model = ResNet18()
#     temp = torch.randn(2, 3, 224, 224)
#     out = model(temp)
#     print("ResNet: ", out.shape)
def test():
    tmp = torch.randn(2, 3, 224, 224)
    temp = torch.randn(2, 3, 224, 224)
    model = ModifiedResNet18(18)
    out = model(tmp, temp)
    print("ResNet: ", out.shape)

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18, self).__init__()
        trained_model1 = resnet18(pretrained=True)
        trained_model2 = resnet18(pretrained=True)
        self.upper_resnet = torch.nn.Sequential(*list(trained_model1.children())[:-3])
        self.lower_resnet = torch.nn.Sequential(*list(trained_model2.children())[:-3])
        self.CrossResidualBlock = CrossResidualBlock(256, 512, stride=2)
        # self.blk1 = ResidualBlock(512, 512, stride=2)
        # self.blk2 = ResidualBlock(512, 512, stride=2)
        # self.CrossResidualBlock1 = CrossResidualBlock(512, 512, stride=2)
        self.fc1 = nn.Linear(512 * 1 * 1, 128)
        self.fc2 = nn.Linear(512 * 1 * 1, 128)
        # self.fc3 = nn.Linear(256 * 1 * 1, 128)
        # self.fc4 = nn.Linear(256 * 1 * 1, 128)
        self.fc8 = nn.Linear(256 * 1 * 1, 128)
        self.fc9 = nn.Linear(256 * 1 * 1, 128)
        self.fc5 = nn.Linear(128 * 1 * 1, 32)
        self.fc6 = nn.Linear(128 * 1 * 1, 32)
        self.fc7 = nn.Linear(32*2, num_classes)

        # netsize = 32
        # self.fc1 = nn.Linear(512, netsize)  # 添加全连接层1
        # #self.dropout1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(512, netsize)  # 添加全连接层2
        # #self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(netsize*2, num_classes) # 合并两个子网络的输出
    def get_layer_output(self, x, layer_name):
        # 通过模型的 forward 方法逐层传递输入数据，直到指定层
        for name, layer in self.named_children():
            x = layer(x)
            if name == layer_name:
                return x
    def fusion(self, x1, x2):
        # size = x1.size(1)
        # num_to_add = size//3
        # x3 = torch.zeros_like(x1)
        # x4 = torch.zeros_like(x2)
        # indices_to_add1 = torch.randperm(size)[:num_to_add]
        # indices_to_add2 = torch.randperm(size)[:num_to_add]
        # x3[:, indices_to_add1] = x1[:, indices_to_add1]
        # x4[:, indices_to_add2] = x2[:, indices_to_add2]
        x3 = x1/3
        x4 = x2/3
        x1 += x4
        x2 += x3
        return x1, x2
    def forward(self, x1, x2):
        # 上面的ResNet
        x1 = self.upper_resnet(x1)

        # 下面的ResNet
        x2 = self.lower_resnet(x2)
        
        x1, x2 = self.CrossResidualBlock(x1, x2)
        # x1, x2 = self.CrossResidualBlock1(x1, x2)
        # print(x1.size())
        # x1 = self.blk1(x1)
        # x2 = self.blk2(x2)
        x1 = F.adaptive_avg_pool2d(x1, [1, 1])
        x2 = F.adaptive_avg_pool2d(x2, [1, 1])
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # x1,x2 = self.fusion(x1, x2)
        # x1 = self.fc1(x1)
        # x2 = self.fc2(x2)
        # combined = torch.cat((x1, x2), dim=1)
        # out = self.fc3(combined)

        x1,x2 = self.fusion(x1, x2)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        # x1,x2 = self.fusion(x1, x2)
        # x1 = self.fc3(x1)
        # x2 = self.fc4(x2)
        # x1,x2 = self.fusion(x1, x2)
        # x1 = self.fc8(x1)
        # x2 = self.fc9(x2)
        x1,x2 = self.fusion(x1, x2)
        x1 = self.fc5(x1)
        x2 = self.fc6(x2)
        combined = torch.cat((x1, x2), dim=1)
        out = self.fc7(combined)


        return out

# test()
