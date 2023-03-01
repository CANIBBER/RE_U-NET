import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from torch.autograd import Function
from torch.autograd import Variable
import torchinfo

class BinarizedF(Function):
  def forward(self, input):
    self.save_for_backward(input)
    a = torch.ones_like(input)
    b = -torch.ones_like(input)
    output = torch.where(input>=0,a,b)
    return output
  def backward(self, output_grad = True):
    input, = self.saved_tensors
    input_abs = torch.abs(input)
    ones = torch.ones_like(input)
    zeros = torch.zeros_like(input)
    input_grad = torch.where(input_abs<=1,ones, zeros)
    return input_grad

class BinarizedModule(nn.Module):
  def __init__(self):
    super(BinarizedModule, self).__init__()
    self.BF = BinarizedF()
  def forward(self,input):
    print(input.shape)
    output =self.BF.forward(input)
    return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class upDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),

        )

    def forward(self, x):
        return self.conv(x)


class SigleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SigleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),

            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class REUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 512],
                 afterfeatures=[64, 64, 128, 256, 512]):
        super(REUNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.sigs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.BW = BinarizedModule()

        for featurenum in range(0, len(features)):
            self.downs.append(DoubleConv(in_channels, afterfeatures[featurenum]))

            in_channels = features[featurenum]
            if featurenum < len(features) - 1:
                self.sigs.append(SigleConv(features[featurenum], features[featurenum + 1]))
        self.downs.append(DoubleConv(512, 512))

        up_features = features[:4]
        up_features = up_features[::-1]
        for featurenum in range(0, len(up_features)):
            if featurenum == 0:
                self.ups.append(
                    upDoubleConv(
                        up_features[featurenum] * 3, up_features[featurenum]
                    )
                )
            else:
                self.ups.append(
                    upDoubleConv(
                        up_features[featurenum] * 5, up_features[featurenum]
                    )
                )

        self.bottleneck = nn.Sequential(
            # nn.MaxPool2d(kernel_size = 2, stride =2),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )
        self.final_conv = nn.Sequential(
            # nn.Conv2d(192, 64, kernel_size = 3, padding=1, stride=1),
            nn.Conv2d(64, 4, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(4, out_channels, kernel_size=1)
        )

    def forward(self, x):
        skip_connections = []
        LOW_connections = []

        for num in range(0, len(self.downs)):

            if num > 1:
                LOW_connections.append(x)
                x = self.sigs[num - 2](x)
            x = self.downs[num](x)
            skip_connections.append(x)
            if num > 0:
                x = self.pool(x)

        #print(x.size())
        x = self.bottleneck(x)
        #print(x.size())
        #print('###')

        skip_connections = skip_connections[::-1]
        LOW_connections = LOW_connections[::-1]

        # for lo in LOW_connections:
        #     print(lo.size())
        # #print('_____')
        # for lo in skip_connections:
        #     print(lo.size())

        for num in range(0, len(self.ups)):
            if num < len(self.ups):
                skip_connection = skip_connections[num]
                LOW_connection = LOW_connections[num]
                connection = torch.cat((skip_connection, LOW_connection), dim=1)
                # print('======')
                # print(connection.size())
                # print(x.size())
                concat_skip = torch.cat((connection, x), dim=1)
            else:
                skip_connection = skip_connections[num]
                # print('*======')
                # print(connection.size())
                # print(x.size())
                concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[num](concat_skip)
            # print("x=")
            # print(x.size())

        x = self.final_conv(x)
        # x = self.BW.forward(x)
        return x


def test():
    x = torch.randn([3, 1, 256, 256])
    B = REUNET(in_channels=1, out_channels=1)
    # print(B)
    pres = B(x)
    assert pres.shape == x.shape
    print(pres.size())
    print(torchinfo.summary(B, input_size=[(3, 1, 256, 256)]))


if __name__ == "__main__":
    test()