## Introduction


### Abstract

In this project, we introduce a deep learning algorithm that uses a 2D RGB image as input and synthesizes a set of multi-focus images.
We use a large light field dataset as the training data. it includes 3343 5D light field data mainly containing plant and flower.
our algorithm is different from the other view synthesize method like light field synthesis or View synthesis by geometry estimation.
we use GAN to capture the semantic relations between different focus level images and synthesis multi-focus images.

<p class="text-center font-weight-bolder"> Test Result </p>

### Background

In fact, we are inspired by a  paper [Learning to Synthesize a 4D RGBD Light Field from a Single Image][5] by Pratul P. Srinivasan.
In this paper, they use a synthesizes algorithm that takes as input a 2D RGB image and synthesizes a 4D RGBD light field. And the synthesizes algorithm contains
two CNN which are to estimate the ray depth and predict occluded rays and non-Lambertian effects. it also include a physical-based function to render a Lambertian approximation
of the light field based on input image and estimated ray depth. 


 
 

 





