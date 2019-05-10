from dataset.config import DatasetConfig, BaseDataset
from torch.utils.data import Dataset
from skimage import transform, io, exposure
import numpy as np
import glob
import os
import cv2


@BaseDataset.register('Flower')
class FlowerPhoto(Dataset, BaseDataset):
    def __init__(self, suffix=('_center', '_1', '_-1'), train=True, size=128):
        img_path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/*_center.png'
        path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/'
        img_list = glob.glob(img_path)
        keys = [p.split('/')[-1][:-11] for p in img_list]
        self.suffix = suffix
        self.img_list = [
            {s[1:]: os.path.join(path, k + s + '.png') for s in suffix} for k in keys
        ]
        if train:
            self.img_list = self.img_list[:int(len(self.img_list) * 0.9)]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * 0.9):]
        self.size = size

    def __getitem__(self, index):
        data = {key: transform.resize(io.imread(val), (self.size, self.size)) for key, val in self.img_list[index].items()}
        for key in data:
            data[key] -= 0.5
            data[key] *= 2
        return data

    def __len__(self):
        return len(self.img_list)


@BaseDataset.register('FlowerFull')
class FlowerFull(Dataset, BaseDataset):
    def __init__(self, train=True, size=256, num=4):
        img_path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/*_center.png'
        path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/'
        self.raw_list = glob.glob(img_path)
        keys = {p.split('/')[-1][:-11]: {'raw': p} for p in self.raw_list if ' ' not in p}

        total_list = glob.glob(path + '*(*).png')
        for p in total_list:
            if ' ' in p:
                continue
            raw_idx = p.split('/')[-1].split('.')[0].split('(')[0]
            series = p.split('/')[-1].split('.')[0].split('(')[1].split(')')[0]
            keys[raw_idx][int(series)] = p

        self.img_list = [v for k, v in keys.items()]

        for p in self.img_list:
            assert len(p) == 5, str(p)

        if train:
            self.img_list = self.img_list[:int(len(self.img_list) * 0.9)]
            # self.raw_list = self.raw_list[:int(len(self.raw_list) * 0.9)]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * 0.9):]
            # self.raw_list = self.raw_list[int(len(self.raw_list) * 0.9):]
        self.size = size

        assert num in (1, 2, 4)
        self.num = num

    def __len__(self):
        return len(self.img_list) * self.num

    def __getitem__(self, item):
        raw_idx = item // self.num
        raw_img = transform.resize(io.imread(self.img_list[raw_idx]['raw']), (self.size, self.size))

        if self.num == 1:
            series_idx = 0
        elif self.num == 2:
            series_idx = 0 if item % self.num == 0 else 3
        else:
            series_idx = item % self.num

        focus_img = transform.resize(io.imread(self.img_list[raw_idx][series_idx]), (self.size, self.size))

        raw_img = exposure.equalize_adapthist(raw_img)
        focus_img = exposure.equalize_adapthist(focus_img)

        raw_img -= 0.5
        raw_img *= 2
        focus_img -= 0.5
        focus_img *= 2
        depth = np.zeros(self.num)
        depth[int(item % self.num)] = 1
        return raw_img, focus_img, depth


@BaseDataset.register('FlowerLF')
class FlowerLF(Dataset, BaseDataset):
    def __init__(self, train=True):
        img_path = '/home/pc2842_columbia_edu/data-disk/Flowers_8bit/*.png'
        self.img_list = glob.glob(img_path)
        if train:
            self.img_list = self.img_list[:int(len(self.img_list) * 0.9)]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * 0.9):]

    @staticmethod
    def load_lf(img_path):
        lfsize = [372, 540]
        # lf = io.imread(img_path)
        lf = cv2.imread(img_path)[:, :, ::-1]
        lf = lf[:lfsize[0] * 14, :lfsize[1] * 14, :3]
        img_shaped = np.reshape(lf, [lfsize[0], 14, lfsize[1], 14, 3])
        img_shaped = img_shaped[:, 3: 10, :, 3: 10, :]
        lff = np.transpose(img_shaped, [1, 3, 0, 2, 4])
        return lff

    def __getitem__(self, index):
        path = self.img_list[index // 49]
        mod = index % 49
        lf = self.load_lf(path) / 255.
        lf *= 2
        lf -= 1
        # 7 7 372 540 3

        center = lf[3, 3]
        center = cv2.resize(center, (256, 256))
        directions = np.zeros((256, 256, 2))
        x = mod // 7
        y = mod % 7
        directions[:, :, 0] = x - 3
        directions[:, :, 1] = y - 3
        target = lf[x, y]
        target = cv2.resize(target, (256, 256))
        return center, directions, target

    def __len__(self):
        return len(self.img_list) * 49
