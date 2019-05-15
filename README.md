# Depth Field GAN

This is the repo for Applied Deep Learning final project. We implemented a GAN that could change the focus location in an image.
To train a model
```bash
python depthfield_gan.py --label_len=1
```
label_len could be {1, 2, 4}.
Label_len corresponds to the length of sigma value in the report.

To evaluate the model, run
```bash
python depthfield_gan_eval.py -- label_len=1
```

The dataset path should be changed before running. You can download the dataset from https://github.com/pratulsrinivasan/Local_Light_Field_Synthesis.
Be prepared because the dataset is very large. To generate ground truth photos for training, use scripts/refocus2.py.

For more information, please visit our website and check out our interactive demo there!