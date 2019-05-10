import glob
import numpy as np
from numba import jit
from skimage import io, transform
import os
import tqdm


lfsize = [372, 540, 8, 8]


@jit
def refocus(lf, shift):
    canvas = np.zeros((372, 540, 3))
    for u in range(7):
        for v in range(7):
            t = transform.EuclideanTransform(translation=((v - 3) * shift, (u - 3) * shift))
            canvas += transform.warp(lf[u, v], t.params)
    return canvas / 49

@jit
def read_files(path):
    lf = io.imread(path)
    lf = lf[:lfsize[0] * 14, :lfsize[1] * 14, :3]
    img_shaped = np.reshape(lf, [lfsize[0], 14, lfsize[1], 14, 3])
    img_shaped = img_shaped[:, 3: 11, :, 3: 11, :]
    lff = np.transpose(img_shaped, [1, 3, 0, 2, 4])
    return lff


if __name__ == '__main__':
    img_paths = glob.glob('/Users/huxin/Desktop/adp/photo/org.png')
    save_path = '/Users/huxin/Desktop/adp/photo/focus/'

    ais = np.arange(0, 2.0, 0.02)
    print(len(ais))
    for path in tqdm.tqdm(img_paths):
        lf = read_files(path)
        file_name = path.split('/')[-1].split('.')[0]
        for ai in ais:
            ai = float("{0:.2f}".format(ai))
            print(ai)
            new_filename = str(int(ai * 50)) + '.png'
            new_img = refocus(lf, ai)
            io.imsave(os.path.join(save_path, new_filename), new_img)
