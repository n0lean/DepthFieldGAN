import torch as th
import abc
import numpy as np
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load
from utils.optim import OptimSchema
from utils.loss import LossSchema


class SamplerConfig(ConfigHelper):
    def __init__(self, name, out_shape, data_num=None, epoch=None,
                 alpha=None, batch_size=None, random_sampling=None,
                 loss_cfg=None, optim_cfg=None, latent_dim=None, max=None):
        super(SamplerConfig, self).__init__()
        self.name = name
        if isinstance(out_shape, tuple):
            out_shape = list(out_shape)
        self.out_shape = out_shape
        self.data_num = data_num
        self.epoch = epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss_cfg = loss_cfg
        self.optim_cfg = optim_cfg
        self.latent_dim = latent_dim
        self.random_sampling = random_sampling
        self.max = max

    def get(self):
        acc_args = BaseSampler.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        return BaseSampler.create(**args)


class BaseSampler(BaseHelper):
    subclasses = {}

    @abc.abstractmethod
    def sampling(self, n):
        pass


class SamplerSchema(Schema):
    name = fields.Str()
    out_shape = fields.List(fields.Int())
    data_num = fields.Int()
    epoch = fields.Int()
    alpha = fields.Float()
    batch_size = fields.Int()
    loss_cfg = fields.Nested(LossSchema)
    optim_cfg = fields.Nested(OptimSchema)
    __model__ = SamplerConfig

    @post_load
    def make_object(self, data):
        return self.__model__(**data)


@BaseSampler.register('Gaussian')
class Gaussian(BaseSampler):
    def __init__(self, out_shape, max=1):
        if isinstance(out_shape, int):
            self.dim = (out_shape,)
        else:
            self.dim = out_shape
        self.max = max

    def sampling(self, n):
        res = th.randn(n, *self.dim) * self.max
        return res


@BaseSampler.register('Uniform')
class Uniform(BaseSampler):
    def __init__(self, out_shape):
        if isinstance(out_shape, int):
            self.dim = (out_shape,)
        else:
            self.dim = out_shape

    def sampling(self, n):
        res = th.rand(n, *self.dim)
        return res


@BaseSampler.register('Onehot')
class Onehot(BaseSampler):
    def __init__(self, out_shape, latent_dim=10, random_sampling=False):
        self.latent_dim = latent_dim
        if isinstance(out_shape, int):
            self.dim = (out_shape,)
        else:
            self.dim = out_shape
        self.random = random_sampling

    def sampling(self, n, with_data=False, dataset=None):
        if self.random:
            return self.sampling_random(n)

        res = np.zeros((n, self.dim[0]))
        data = []
        for i in range(n):
            t = np.random.randint(0, self.dim[0])
            if with_data:
                data.append(dataset[t])
            res[i, t] = 1
        res = th.Tensor(res)
        if with_data:
            data = th.Tensor(np.array(data))
            return res, data
        return res

    def sampling_random(self, n):
        res = th.randn(n, *self.dim)
        res = res / th.sum(res ** 2) ** 0.5
        return res

    def sampling_with_dataset(self, n, data):
        res = np.zeros((n, self.dim[0]))
        data_partial = []
        for i in range(n):
            idx = np.random.randint(0, self.dim[0])
            res[i, idx] = 1
            data_item = data[idx]
            data_partial.append(data_item)
        data_partial = np.array(data_partial)
        res = th.Tensor(res)
        data_partial = th.Tensor(data_partial)
        return res, data_partial


@BaseSampler.register('Learning')
class Learning(BaseSampler):
    def __init__(self, out_shape, data_num, epoch, alpha, batch_size, loss_cfg, optim_cfg):
        if isinstance(out_shape, int):
            self.dim = (out_shape,)
        else:
            self.dim = out_shape
        self.data_num = data_num
        self.sampler_modes = th.tensor.randn(data_num, *self.dim)
        self.scale = th.ones(data_num, *self.dim)
        self.epoch = epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg

        self.optim = self.optim_cfg.get([self.sampler_modes.requires_grad_(), self.scale.requires_grad_()])

    def sampling(self, n):
        idx = np.random.randint(0, self.data_num, n)
        scale = self.scale[idx]
        res = th.randn(n, *self.dim) * scale
        base_res = self.sampler_modes[idx]
        return res + base_res

    def update(self, gan):
        gen = gan.gen
        gan.eval_mode()

        total_batch = self.data_num // self.batch_size + int(self.data_num % self.batch_size > 0)
        cri = self.loss_cfg.get()
        device = gan.device

        self.sampler_modes.to(device)
        self.scale.to(device)

        target_data = gan.dataset.data
        target_data = th.Tensor(target_data).to(device).detach()
        for ep in range(self.epoch):
            for b_num in range(total_batch):
                if b_num != total_batch - 1:
                    idx = np.arange(b_num * self.batch_size, (b_num + 1) * self.batch_size)
                else:
                    idx = np.arange(b_num * self.batch_size, self.data_num)
                samples = self.sampler_modes[idx].to(device)
                scale = self.scale[idx].to(device)
                target = target_data[idx]
                self.optim.zero_grad()
                noise = th.randn(*samples.shape, requires_grad=False, device=device)
                out = gen(samples + scale * noise)
                loss = cri(out, target)
                loss.backward(retain_graph=True)
                self.optim.step()


@BaseSampler.register('Learning')
class Learning(BaseSampler):
    def __init__(self, out_shape, data_num, epoch, alpha, batch_size, loss_cfg, optim_cfg):
        if isinstance(out_shape, int):
            self.dim = (out_shape,)
        else:
            self.dim = out_shape
        self.data_num = data_num
        self.sampler_modes = th.randn(data_num, *self.dim)
        self.scale = th.ones(data_num, *self.dim)
        self.epoch = epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg

        self.optim = self.optim_cfg.get([self.sampler_modes.requires_grad_(), self.scale.requires_grad_()])

    def sampling(self, n):
        idx = np.random.randint(0, self.data_num, n)
        scale = self.scale[idx]
        res = th.randn(n, *self.dim) * scale
        base_res = self.sampler_modes[idx]
        return res + base_res

    def update(self, gan, recon_ratio=1, dis_ratio=0):
        gen = gan.gen
        gan.eval_mode()

        total_batch = self.data_num // self.batch_size + int(self.data_num % self.batch_size > 0)
        cri = self.loss_cfg.get()
        device = gan.device

        self.sampler_modes.to(device)
        self.scale.to(device)

        target_data = gan.dataset.data
        target_data = th.Tensor(target_data).to(device).detach()
        for ep in range(self.epoch):
            for b_num in range(total_batch):
                if b_num != total_batch - 1:
                    idx = np.arange(b_num * self.batch_size, (b_num + 1) * self.batch_size)
                else:
                    idx = np.arange(b_num * self.batch_size, self.data_num)
                samples = self.sampler_modes[idx].to(device)
                scale = self.scale[idx].to(device)
                target = target_data[idx]
                self.optim.zero_grad()
                noise = th.randn(*samples.shape, requires_grad=False, device=device)
                out = gen(samples + scale * noise)
                loss = cri(out, target)
                loss.backward(retain_graph=True)
                self.optim.step()