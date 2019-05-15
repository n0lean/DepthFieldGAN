### Background

In fact, we are inspired by a paper [Learning to Synthesize a 4D RGBD Light Field from a Single Image][5] by Pratul P. Srinivasan.
In this paper, they use a synthesizes algorithm that takes as input a 2D RGB image and synthesizes a 4D RGBD light field. And the synthesizes algorithm contains
two CNN which are to estimate the ray depth and predict occluded rays and non-Lambertian effects. it also include a physical-based function to render a Lambertian approximation
of the light field based on input image and estimated ray depth. 
