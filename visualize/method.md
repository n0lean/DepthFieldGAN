## Method

#### Data generation and pre-processing

In order to obtain pairs of images with different level of focus,
we need to make use of a special camera named light field camera.

"A light field camera, also known as plenoptic camera, captures information about the light field emanating from a scene; that is, the intensity of light in a scene, and also the direction that the light rays are traveling in space. "

In general, images captured by the camera is a 5D array which is a 2D array of images.

With its help, we could get different level focused images by the 
following equation:

<img src="https://s3.us-east-2.amazonaws.com/n0lean-bucket/lf_eq.png"  class="rounded mx-auto d-block" alt="Responsive image" width=30%/>

Then we generated a series of near-focus and far-focus images, which is also called focus stack.

<p class="text-center font-weight-bolder"> Focus Stack </p>
We implemented above by following code:

----

```python
@jit
def refocus(lf, shift):
    canvas = np.zeros((372, 540, 3))
    for u in range(7):
        for v in range(7):
            t = transform.EuclideanTransform(translation=((v - 3) * shift, (u - 3) * shift))
            canvas += transform.warp(lf[u, v], t.params)
    return canvas / 49
```

----

One way to artificially refocus is to synthesis light field. Through synthesis light filed is promising, 
the computational cost is huge and the algorithm is extreme complex.
Let the input image size be W\*H\*C, the model needs to output a u\*v\*W\*H\*C light field
as well as a same size depth prediction. 
Generative Adversarial Network (GAN) is a new approach in image 
synthesis. We assume GAN could capture the semantic relations between different focus level images.
In order to achieve this, we used following architecture and training strategy.


#### GAN architecture

<img src="https://s3.us-east-2.amazonaws.com/n0lean-bucket/gen.png" class="rounded mx-auto d-block" alt="Responsive image" width=80%/>


We use the architecture similar to [Pix2Pix][3]. For generator, We used a
UNet implementation in Generator instead of a ResNet implementation (as you can see in the following code snippet), because we found the direct concatenation of low level
feature with transposed convolution output would benefit the reconstruction process. Additionally, we used a hyper-parameter 
sigma to control the focus level as suggested by [RefocusGAN][2]. Two fully connected layers is used to to transform sigma to 
a 4x4 feature map then it is concatenated to inner most block's output. 

----
```python 
# in model/zoo.py

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
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
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
                                             norm_layer=norm_layer, use_resizeconv=use_resizeconv)  # add the outermost layer
        self.dense = nn.Linear(label_len, 4 * 4)

    def forward(self, x, x1):
        """Standard forward"""
        x1 = self.dense(x1).view(-1, 1, 4, 4)
        return self.model(x, x1)
```
----

Unlike open source implementation of Pix2Pix, we do not use transposed convolution. Instead, 
we use upsampling with convolution as suggested by [Odena, et al.][2] to avoid checkerboard artifacts.
Code for building upsampling block:

----
```python
# in model/zoo.py

if use_resizeconv: # use our implemented upsampling and convolution
    upconv = [
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, stride=1)
    ]
    down = [downconv]
    up = [uprelu, *upconv, nn.Tanh()]
    model = down + [submodule] + up
else: # use native transposed convolution
    upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                kernel_size=4, stride=2,
                                padding=1)
    down = [downconv]
    up = [uprelu, upconv, nn.Tanh()]
    model = down + [submodule] + up
```
----

As for discriminator, we used a similar fashion as PatchGAN. Instead of judging the whole image, the discriminator judge 
every patch (70x70 in our case) whether is fake or true.

<img src="https://s3.us-east-2.amazonaws.com/n0lean-bucket/dis.png" class="rounded mx-auto d-block"  alt="Responsive image" width=60%/>

----

```python
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
```
----
We use [LSGAN][5]'s loss, because MSE loss could make the training more stable. We also use [perceptual loss][4] as part of generator loss. The
perceptual loss could encourage generator generates more realistic samples including high frequency region which is hard to 
achieve using a L2 Loss.

<img src="https://s3.us-east-2.amazonaws.com/n0lean-bucket/ploss.png" class="rounded mx-auto d-block"  alt="Responsive image" width=50%/>

#### Train Details

Both learning rates for generator and discriminator are
2e-4. Batch size is 12. We follow the same learning rate scheduling method as Pix2Pix which uses 2e-4 for the first 100 epochs
. And reduce learning rate to 0 linearly in the last 100 epochs. We also set beta0 to 0.5 for Adam. The total training time is about 4 days on a Nvidia K80.
The model is implemented in PyTorch.
----
