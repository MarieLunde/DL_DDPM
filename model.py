from torch import nn

class DummyUnet(nn.Module):
    def __init__(self, image_size, channels):
        """
        image_size: the heigth/width of the square image
        channels: 1 for MNIST and 3 for CIFAR        
        """
        super(DummyUnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.Conv2d(in_channels=64,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding='same')
        )
    def forward(self, x):
        return self.net(x)
    

class Unet():
    pass