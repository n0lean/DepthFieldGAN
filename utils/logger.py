import os
import datetime
import json
import logging
from skimage import io
import numpy as np


BASE_DIR = './log'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


class Logger(object):
    def __init__(self, cat):
        self.cat = cat
        self.id = self._init_dir_()
        self.exp_dir = os.path.join(BASE_DIR, self.cat, self.id)
        self.img_dir = os.path.join(BASE_DIR, self.cat, self.id, 'img')
        self.scalar = {}

    def _init_dir_(self):
        if not os.path.exists(os.path.join(BASE_DIR, self.cat)):
            os.makedirs(os.path.join(BASE_DIR, self.cat))
        timestamp = str(datetime.datetime.now()).replace(' ', '')
        os.makedirs(os.path.join(BASE_DIR, self.cat, timestamp, 'img'))
        return timestamp

    def save_cfg(self, cfg):
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f)

    def create_scalar(self, name):
        self.scalar[name] = []

    def append_scalar(self, name, new_scalar):
        self.scalar[name].append(new_scalar)

    def save_scalar(self):
        with open(os.path.join(self.exp_dir, 'scalar.json'), 'w') as f:
            json.dump(self.scalar, f)

    def log_img(self, img_id, data, func=None, **kwargs):
        if func is not None:
            data = func(data, **kwargs)
        io.imsave(os.path.join(self.img_dir, img_id), data)

    def save_npy(self, name, obj):
        assert isinstance(obj, np.ndarray)
        np.save(os.path.join(self.exp_dir, name + '.npy'), obj)
