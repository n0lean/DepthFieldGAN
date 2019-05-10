import torch as th
from torch import nn
from torch.nn import functional as F
from utils.class_helper import BaseHelper, ConfigHelper
from marshmallow import Schema, fields, post_load
import functools
import numpy as np
import math


class ModelConfig(ConfigHelper):
    def __init__(self, name, **kwargs):
        super(ModelConfig, self).__init__()
        self.name = name
        self.__dict__.update(kwargs)

    def get(self):
        acc_args = BaseModel.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        return BaseModel.create(**args)


class BaseModel(BaseHelper):
    subclasses = {}


@BaseModel.register('MLG')
class MultiLayerGenerator(BaseModel, nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerGenerator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.5)
        x = F.leaky_relu(self.map2(x), 0.5)
        return self.map3(x)


@BaseModel.register('MLD')
class MultiLayerDiscriminator(BaseModel, nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sigmoid=True):
        super(MultiLayerDiscriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.5)
        x = F.leaky_relu(self.map2(x), 0.5)
        x = self.map3(x)
        if self.sigmoid:
            x = F.sigmoid(x)
        return x



@BaseModel.register('DCG')
class DCGenerator(BaseModel, nn.Module):
    def __init__(self, input_size=100, hidden_size=128, output_size=3, instance_norm=True):
        super(DCGenerator, self).__init__()
        self.instance_norm = instance_norm
        self.deconv1 = nn.ConvTranspose2d(input_size, hidden_size * 8, 4, 1, 0)
        self.deconv1_bn = nn.InstanceNorm2d(hidden_size * 8) if instance_norm else nn.BatchNorm2d(hidden_size * 8)
        self.deconv2 = nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1)
        self.deconv2_bn = nn.InstanceNorm2d(hidden_size * 4) if instance_norm else nn.BatchNorm2d(hidden_size * 4)
        self.deconv3 = nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1)
        self.deconv3_bn = nn.InstanceNorm2d(hidden_size * 2) if instance_norm else nn.BatchNorm2d(hidden_size * 2)
        self.deconv5 = nn.ConvTranspose2d(hidden_size * 2, output_size, 4, 2, 1)

    def forward(self, inputs):
        x = F.relu(self.deconv1_bn(self.deconv1(inputs)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv5(x))
        return x


@BaseModel.register('DCD')
class DCDiscriminator(BaseModel, nn.Module):
    def __init__(self, input_size=3, hidden_size=128, out_64=False, sigmoid=True):
        super(DCDiscriminator, self).__init__()
        self.out_64 = out_64
        if out_64:
            self.conv1 = nn.Conv2d(input_size, hidden_size, 4, 2, 1)
            self.conv1_bn = nn.BatchNorm2d(hidden_size)
            self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(hidden_size * 2)
        else:
            self.conv2 = nn.Conv2d(input_size, hidden_size * 2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(hidden_size * 2)
        self.conv3 = nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(hidden_size * 4)
        self.conv4 = nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(hidden_size * 8)
        self.conv5 = nn.Conv2d(hidden_size * 8, 1, 4, 1, 0)
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.out_64:
            x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        if self.sigmoid:
            x = F.sigmoid(x)
        return x


@BaseModel.register('FeatModel')
class FeatModel(BaseModel, nn.Module):
    def __init__(self, input_nc=3, hidden_size=32, out_nc=100, sigmoid=True):
        super(FeatModel, self).__init__()
        self.out_nc = out_nc
        self.conv1 = nn.Conv2d(input_nc, hidden_size, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(hidden_size * 2)
        self.conv3 = nn.Conv2d(hidden_size * 2, out_nc, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(out_nc)
        self.global_pool = nn.AvgPool2d(4)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.global_pool(x)
        # x = x / th.norm(x, p=2)
        x = x.view(-1, self.out_nc, 1, 1)
        if self.sigmoid:
            x = th.sigmoid(x)
        return x



@BaseModel.register('ResNetGen')
class ResNetGen(BaseModel, nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_bias=True, n_blocks=9):
        norm_layer = nn.InstanceNorm2d
        use_dropout = False
        padding_type = 'reflect'

        super(ResNetGen, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


@BaseModel.register('PatchDis')
class PatchDis(nn.Module, BaseModel):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_bias=True):
        norm_layer = nn.BatchNorm2d

        super(PatchDis, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


@BaseModel.register('UNetGen')
class UnetGenerator(nn.Module, BaseModel):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_resizeconv=False, ex_label=True, label_len=2):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True, use_resizeconv=use_resizeconv,
                                             ex_label=ex_label)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, use_resizeconv=use_resizeconv,
                                                 submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,
                                                 ex_label=True)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_resizeconv=use_resizeconv, ex_label=True)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, ex_label=True,
                                             submodule=unet_block, norm_layer=norm_layer, use_resizeconv=use_resizeconv)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, ex_label=True,
                                             norm_layer=norm_layer, use_resizeconv=use_resizeconv)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, ex_label=True,
                                             submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer,
                                             use_resizeconv=use_resizeconv)  # add the outermost layer
        self.dense = nn.Linear(label_len, 4 * 4)

    def forward(self, x, x1):
        """Standard forward"""
        x1 = self.dense(x1).view(-1, 1, 4, 4)
        return self.model(x, x1)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_resizeconv=False, ex_label=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.ex_label = ex_label
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if use_resizeconv:
                upconv = [
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, stride=1)
                ]
                down = [downconv]
                up = [uprelu, *upconv, nn.Tanh()]
                model = down + [submodule] + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
        elif innermost:
            if use_resizeconv:
                upconv = [
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc + 1, outer_nc, kernel_size=3, padding=1, stride=1)
                ]
                down = [downrelu, downconv]
                up = [uprelu, *upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc + 1, outer_nc,
                                            kernel_size=3, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
        else:
            if use_resizeconv:
                upconv = [
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, stride=1)
                ]
                down = [downrelu, downconv, downnorm]
                up = [uprelu, *upconv, upnorm]
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        if self.ex_label:
            self.down_model = nn.Sequential(*down)
            self.up_model = nn.Sequential(*up)
            self.submodule = submodule
        else:
            self.model = nn.Sequential(*model)

    def forward(self, x, x1):
        if self.ex_label:
            if self.innermost:
                inner = self.down_model(x)
                inner = th.cat([inner, x1], 1)
                outer = self.up_model(inner)
                return th.cat([x, outer], 1)
            elif self.outermost:
                inner = self.down_model(x)
                sub = self.submodule(inner, x1)
                outer = self.up_model(sub)
                return outer
            else:
                inner = self.down_model(x)
                sub = self.submodule(inner, x1)
                outer = self.up_model(sub)
                return th.cat([x, outer], 1)
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return th.cat([x, self.model(x)], 1)


@BaseModel.register('PixelDis')
class PixelDiscriminator(nn.Module, BaseModel):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


