"""
The model file of the Progressive GAN
"""
# import all libraries
import torch
from torch import nn
import torch.nn.functional as F
from math import log2

# define factors
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    """
    Weight scaled Conv2d block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        # define the conv layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # define the new bias and the weight scale
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """
    Pixel normalization 
    """
    def __init__(self):
        super(PixelNorm, self).__init__()
        # define epsilon
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):
    """
    Define a Convolution block. This block uses:
        1. Weight scaled Conv
        2. Pixel normaliztion
        3. leaky relu
    """
    def __init__(self, in_channels, out_channels, use_pixelnorm = True):
        super(ConvBlock, self,).__init__()
        self.use_pixelnorm = use_pixelnorm
        # define WSConvs
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        # leaky relu with the slope of 2
        self.leaky_relu = nn.LeakyReLU(0.2)
        # pixel normalization layer
        self.pixelnorm = PixelNorm()

    def forward(self, x):
        """
        First use the conv1 with the leaky relu activation,
        then apply a pixelnorm. 
        After that use conv2 with leaky relu function and again apply pixel norm
        """
        x = self.leaky_relu(self.conv1(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x
        x = self.leaky_relu(self.conv2(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x

        return x

class Generator(nn.Module):
    """
    The progressive generator model 
    """
    def __init__(self, z_dim, in_channels, img_channels= 3):
        super(Generator, self).__init__()
        # define the initial, it takes 1x1 to 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        # define module lists for prog blocks and rgb layers
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        # progressive 
        for i in range(len(factors) - 1):
            # get input and output channels
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            # add conv layers to prog blocks and rgb layers lists
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))
    
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
            out = self.initial(x)

            # if it was step 0, return the initial rgb layer 
            if steps == 0:
                return self.initial_rgb(out)

            for step in range(steps):
                upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
                out = self.prog_blocks[step](upscaled)

            
            # final upscale
            final_upscaled = self.rgb_layers[steps - 1](upscaled)
            final_out = self.rgb_layers[steps](out)

            return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    """
    Progressive discriminator model
    Because Discriminator is mirrored of the Gen model, 
    The final layer is for 4x4
    """
    def __init__(self, z_dim, in_channels, img_channels):
        super(Discriminator, self).__init__()
        # define lists for progressive blocks and rgb layers
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([])

        # define the leaky relu function with the slope of 0.2
        self.leaky_relu = nn.LeakyReLU(0.2)

        # progressive part
        for i in range(len(factors) - 1, 0, -1):
            # get input and output channels for conv layers
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])

            # add blocks and layers to their dedicated lists
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size = 1, stride=1, padding=0))

        # define the initial RGB layer (it's for 4x4)
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)

        # define average pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # block for 4x4 
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        # define batch statistics
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))

        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        # get current step
        current_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step
        out = self.leaky_relu(self.rgb_layers[current_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # do downscaling (use RGB)
        downscaled = self.leaky_relu(self.rgb_layers[current_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[current_step](out))

        # perform fade in
        out = self.fade_in(alpha, downscaled, out)

        for step in range(current_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        # minibatch
        out = self.minibatch_std(out)

        return self.final_block(out).view(out.shape[0], -1)

