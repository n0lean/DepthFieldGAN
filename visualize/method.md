
<br>
<br>

### Method

#### Data generation and pre-processing

In order to obtain pairs of images with different level of focus,
we need to make use of a special camera named light field camera.
"A light field camera, also known as plenoptic camera, captures information about the light field emanating from a scene; that is, the intensity of light in a scene, and also the direction that the light rays are traveling in space. "
In general, images captured by the camera is a 5D array which is a 2D array of images.
With its help, we could get different level focused images by the following equation:

<img src="https://s3.us-east-2.amazonaws.com/n0lean-bucket/lf_eq.png"  class="rounded mx-auto d-block" alt="Responsive image" width=30%/>

Then we generated a series of near-focus and far-focus images, which is also called focus stack.
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
Let the input image size be W\*H\*C, the model needs to output a Uu\*U\*W\*H\*C light field
as well as a same size depth prediction. 
Generative Adversarial Network (GAN) is a new approach in image 
synthesis. We assume GAN could capture the semantic relations between different focus level images.
In order to achieve this, we used following architecture and training strategy.

Here is a list of images with different focus level from the same focus stack.

<p class="text-center font-weight-bolder"> Focus Stack </p>