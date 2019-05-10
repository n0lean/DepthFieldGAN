import abc
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load
import torch as th
from utils.loss import *
from utils.optim import *
from utils.sampler import *
from utils.dataloader import *
from model.zoo import *
from dataset.config import *
import tqdm
import os
from utils.image_pool import ImagePool
import itertools


class GanConfig(ConfigHelper):
    def __init__(self, name, **kwargs):
        super(GanConfig, self).__init__()
        self.name = name
        self.__dict__.update(kwargs)

    def get(self):
        acc_args = BaseGan.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        args['device'] = th.device(args['device'])
        return BaseGan.create(**args)


# class GanSchema(Schema):
#     name = fields.Str()
#
#     gen_cfg = fields.Nested(ModelSchema)
#     dis_cfg = fields.Nested(ModelSchema)
#
#     gan_epoch = fields.Int()
#
#     dataset_cfg = fields.Nested(DatasetSchema)
#     loader_cfg = fields.Nested(LoaderSchema)
#
#     gloss_cfg = fields.Nested(LossSchema)
#     dloss_cfg = fields.Nested(LossSchema)
#
#     goptim_cfg = fields.Nested(OptimSchema)
#     doptim_cfg = fields.Nested(OptimSchema)
#
#     sampler_cfg = fields.Nested(SamplerSchema)
#     label_smooth = fields.Bool()
#     device = fields.Str()


class BaseGan(BaseHelper):
    _pre_iterate_hooks_ = []
    _post_iterate_hooks_ = []
    _mid_iterate_hooks_ = []

    _pre_train_hooks_ = []
    _post_train_hooks_ = []

    subclasses = {}

    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 sampler_cfg, label_smooth, dist_loss,
                 device):
        self.device = device
        self.gen = gen_cfg.get().to(self.device)
        self.dis = dis_cfg.get().to(self.device)
        self.gen_step = gen_step
        self.dis_step = dis_step
        self.gan_epoch = gan_epoch
        self.dataset = dataset_cfg.get()
        self.dataloader = loader_cfg.get(self.dataset)
        self.batch_size = loader_cfg.batch_size
        self.goptim_cfg = goptim_cfg
        self.doptim_cfg = doptim_cfg
        self.gen_loss = gloss_cfg.get()
        self.dis_loss = dloss_cfg.get()
        self.gen_optim = goptim_cfg.get(self.gen.parameters())
        self.dis_optim = doptim_cfg.get(self.dis.parameters())
        self.sampler = sampler_cfg.get()
        self.label_smooth = label_smooth
        self.dist_loss = dist_loss
        self.z_dist = th.nn.MSELoss()
        self.g_dist = th.nn.MSELoss()
        self.criterion_mse = th.nn.MSELoss()
        self.alpha = 0.01

    def train_mode(self):
        self.gen.train()
        self.dis.train()

    def eval_mode(self):
        self.gen.eval()
        self.dis.eval()

    def save(self, path, name):
        if not os.path.exists(os.path.join(path, name)):
            os.makedirs(os.path.join(path, name))
        state_dict = self.gen.state_dict()
        th.save(state_dict, os.path.join(path, name, 'gen.pth.tar'))
        state_dict = self.dis.state_dict()
        th.save(state_dict, os.path.join(path, name, 'dis.pth.tar'))

    def load(self, path):
        state_dict = th.load(os.path.join(path, 'gen.pth.tar'))
        self.gen.load_state_dict(state_dict)
        state_dict = th.load(os.path.join(path, 'dis.pth.tar'))
        self.dis.load_state_dict(state_dict)

    def gen_iter(self):
        self.gen.zero_grad()
        latent_samples = self.sampler.sampling(self.batch_size)
        g_gen_input = latent_samples.to(dtype=th.float32, device=self.device)
        g_fake_data = self.gen(g_gen_input)
        g_fake_deision = self.dis(g_fake_data)
        g_fake_labels = th.ones(g_fake_deision.shape, dtype=th.float32, device=self.device)

        g_loss = self.gen_loss(g_fake_deision, g_fake_labels)

        if self.dist_loss:
            latent_samples2 = self.sampler.sampling(self.batch_size)
            g_gen_input2 = latent_samples2.to(dtype=th.float32, device=self.device)
            g_fake_data2 = self.gen(g_gen_input2)

            z_dist = self.z_dist(g_gen_input, g_gen_input2)
            g_dist = self.g_dist(g_fake_data, g_fake_data2)

            dist_loss = self.criterion_mse(th.log(g_dist).to(self.device),
                                           100 * th.log(z_dist).to(self.device))
            g_loss += self.alpha * dist_loss
        g_loss.backward()
        self.gen_optim.step()

    def dis_iter(self):
        self.dis.zero_grad()
        real_samples = next(iter(self.dataloader))
        if isinstance(real_samples, list):
            real_samples = real_samples[0]
        d_real_data = real_samples.to(dtype=th.float32, device=self.device)
        d_real_decision = self.dis(d_real_data)
        d_real_labels = th.ones(d_real_decision.shape, dtype=th.float32, device=self.device)
        d_real_loss = self.dis_loss(d_real_decision, d_real_labels)
        if self.label_smooth:
            d_real_loss += F.kl_div(
                d_real_decision,
                th.ones(d_real_decision.shape, dtype=th.float32, device=self.device) * 0.5
            )

        latent_samples = self.sampler.sampling(self.batch_size)
        d_gen_input = latent_samples.to(dtype=th.float32, device=self.device)
        d_fake_data = self.gen(d_gen_input)
        d_fake_decision = self.dis(d_fake_data)
        d_fake_labels = th.zeros(d_fake_decision.shape, dtype=th.float32, device=self.device)
        d_fake_loss = self.dis_loss(d_fake_decision, d_fake_labels)

        if self.label_smooth:
            d_fake_loss += F.kl_div(
                d_fake_decision,
                th.ones(d_fake_decision.shape, dtype=th.float32, device=self.device) * 0.5
            )

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.dis_optim.step()

    def iterate(self, gen_step=None, dis_step=None, **kwargs):
        self.train_mode()
        gen_step = gen_step if gen_step is not None else self.gen_step
        dis_step = dis_step if dis_step is not None else self.dis_step

        for f in self._pre_iterate_hooks_:
            f(self, **kwargs)

        for _ in range(dis_step):
            self.dis_iter()

        for f in self._mid_iterate_hooks_:
            f(self, **kwargs)

        for _ in range(gen_step):
            self.gen_iter()

        for f in self._post_iterate_hooks_:
            f(self, **kwargs)

    def train(self, epochs=None, gen_step=None, dis_step=None, use_tqdm=False, **kwargs):
        epochs = epochs if epochs is not None else self.gan_epoch
        gen_step = gen_step if gen_step is not None else self.gen_step
        dis_step = dis_step if dis_step is not None else self.dis_step
        pbar = tqdm.tqdm(range(epochs)) if use_tqdm else range(epochs)
        for f in self._pre_train_hooks_:
            f(self, **kwargs)

        for iter_num in pbar:
            self.iterate(gen_step, dis_step, iter_num=iter_num)

        for f in self._post_train_hooks_:
            f(self, **kwargs)

    def sampling(self, num_batch, batch_size):
        generated_pts = []
        for i in range(num_batch):
            samples = self.sampler.sampling(batch_size).to(self.device).float()
            self.eval_mode()
            res = self.gen(samples)
            generated_pts.append(res.cpu().detach().numpy())
        generated_pts = np.concatenate(generated_pts)
        return generated_pts

    @classmethod
    def add_pre_iterate_hook(cls, func):
        cls._pre_iterate_hooks_.append(func)

    @classmethod
    def add_post_iterate_hook(cls, func):
        cls._post_iterate_hooks_.append(func)

    @classmethod
    def add_mid_iterate_hook(cls, func):
        cls._mid_iterate_hooks_.append(func)

    @classmethod
    def add_pre_train_hook(cls, func):
        cls._pre_train_hooks_.append(func)

    @classmethod
    def add_post_train_hook(cls, func):
        cls._post_train_hooks_.append(func)


@BaseGan.register('WGAN-GP')
class WGANGP(BaseGan):
    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 sampler_cfg, label_smooth, dist_loss,
                 device, LAMBDA):
        super(WGANGP, self).__init__(gen_cfg, dis_cfg, gen_step,
                                     dis_step, gan_epoch, loader_cfg,
                                     dataset_cfg, gloss_cfg, dloss_cfg,
                                     goptim_cfg, doptim_cfg,
                                     sampler_cfg, label_smooth, dist_loss,
                                     device)
        self.LAMBDA = LAMBDA
        self.one = th.tensor(1.0).to(device)
        self.mone = self.one * -1
        self.mone.to(device)

    def gen_iter(self):
        self.gen.zero_grad()
        latent_samples = self.sampler.sampling(self.batch_size)
        g_gen_input = latent_samples.to(dtype=th.float32, device=self.device)
        g_fake_data = self.gen(g_gen_input)
        g_fake_deision = self.dis(g_fake_data)

        g_loss = g_fake_deision.mean()
        g_loss.backward(self.mone)
        self.gen_optim.step()

    def dis_iter(self):
        self.dis.zero_grad()
        real_samples = next(iter(self.dataloader))
        if isinstance(real_samples, list):
            real_samples = real_samples[0]
        d_real_data = real_samples.to(dtype=th.float32, device=self.device)
        d_real_decision = self.dis(d_real_data)
        d_real_loss = d_real_decision.mean()
        d_real_loss.backward(self.mone)

        latent_samples = self.sampler.sampling(self.batch_size)
        d_gen_input = latent_samples.to(dtype=th.float32, device=self.device)
        d_fake_data = self.gen(d_gen_input)
        d_fake_decision = self.dis(d_fake_data)
        d_fake_loss = d_fake_decision.mean()
        d_fake_loss.backward(self.one)

        gp_fake_latent = self.sampler.sampling(self.batch_size).to(dtype=th.float32, device=self.device)
        gp_fake_data = self.gen(gp_fake_latent)
        gp_real_samples = next(iter(self.dataloader))
        if isinstance(gp_real_samples, list):
            gp_real_samples = gp_real_samples[0]
        gp_real_data = gp_real_samples.to(dtype=th.float32, device=self.device)
        gp = self.calc_gradient_penalty(gp_real_data, gp_fake_data, self.LAMBDA)
        gp.backward()

        d_loss = d_fake_loss - d_real_loss + gp
        self.dis_optim.step()

    def calc_gradient_penalty(self, real_data, fake_data, LAMBDA):
        alpha = th.rand(self.batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(self.device)
        disc_interpolates = self.dis(interpolates)
        gradients = th.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                     grad_outputs=th.ones(disc_interpolates.size()).to(self.device),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty



