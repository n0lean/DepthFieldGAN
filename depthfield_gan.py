import matplotlib
matplotlib.use('agg')

from utils.sampler import SamplerConfig
from utils.loss import LossConfig
from utils.optim import OptimConfig
from utils.dataloader import LoaderConfig
from model.gan import GanConfig, BaseGan
from model.zoo import ModelConfig
from dataset.config import DatasetConfig
from matplotlib import pyplot as plt
from utils.util_model import UtilityModelConfig, UtilityModelSchema
import os
from torchvision.utils import make_grid, save_image
import numpy as np
import torch as th
from torch import nn
from tensorboardX import SummaryWriter
from torchvision import models
import argparse


@BaseGan.register('Pix2Pix')
class Pix2Pix(BaseGan):
    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 device):
        self.device = device
        self.gen = gen_cfg.get().to(self.device)
        self.dis = dis_cfg.get().to(self.device)

        self.gen_loss = gloss_cfg.get()
        self.dis_loss = dloss_cfg.get()
        self.goptim_cfg = goptim_cfg
        self.doptim_cfg = doptim_cfg

        self.gen_step = gen_step
        self.dis_step = dis_step
        self.gan_epoch = gan_epoch
        self.dataset = dataset_cfg.get()
        self.dataloader = loader_cfg.get(self.dataset)
        self.batch_size = loader_cfg.batch_size

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gen_optim = self.goptim_cfg.get(self.gen.parameters())
        self.dis_optim = self.doptim_cfg.get(self.dis.parameters())

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - GAN_EPOCHS + DECREASE_LR_EPOCHS) / DECREASE_LR_EPOCHS
            return lr_l

        self.gen_scheduler = th.optim.lr_scheduler.LambdaLR(self.gen_optim, lr_lambda=lambda_rule)
        self.dis_scheduler = th.optim.lr_scheduler.LambdaLR(self.dis_optim, lr_lambda=lambda_rule)

        self.lambda_L1 = 100

        vgg = models.vgg16(pretrained=True).to(device)
        self.vgg_feat = nn.Sequential(*list(vgg.children())[:-4])
        for param in self.vgg_feat.parameters():
            param.requires_grad = False
        self.vgg_feat.eval()

    def iterate(self, gen_step=None, dis_step=None, **kwargs):
        self.train_mode()
        gen_step = gen_step if gen_step is not None else self.gen_step
        dis_step = dis_step if dis_step is not None else self.dis_step
        data = next(iter(self.dataloader))
        self.real_a, self.real_b, self.depth = data

        self.depth = self.depth.to(self.device).float()
        self.real_a = self.real_a.permute((0, 3, 1, 2)).to(self.device).float()
        self.real_b = self.real_b.permute((0, 3, 1, 2)).to(self.device).float()

        self.real_a_depth = [self.real_a, self.depth]

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

    def gen_iter(self):
        self.gen.zero_grad()

        self.fake_b = self.gen(x=self.real_a_depth[0], x1=self.real_a_depth[1])

        fake_vgg_feat = self.vgg_feat(self.fake_b)
        true_vgg_feat = self.vgg_feat(self.real_b)

        self.perceptual_loss = self.l1_loss(fake_vgg_feat, true_vgg_feat)

        fake_ab = th.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.dis(fake_ab)
        g_fake_labels = th.ones(pred_fake.shape, dtype=th.float32, device=self.device)
        self.g_fake_loss = self.gen_loss(pred_fake, g_fake_labels)
        # self.g_loss = g_fake_loss + self.lambda_L1 * self.l1_loss(self.fake_b, self.real_b)
        self.g_loss = self.g_fake_loss + self.lambda_L1 * self.perceptual_loss
        self.g_loss.backward()
        self.gen_optim.step()

    def dis_iter(self):
        self.dis.zero_grad()
        self.fake_b = self.gen(x=self.real_a_depth[0], x1=self.real_a_depth[1])
        fake_ab = th.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.dis(fake_ab)

        real_ab = th.cat((self.real_a, self.real_b), 1)
        pred_real = self.dis(real_ab)

        d_fake_labels = th.zeros(pred_fake.shape, dtype=th.float32, device=self.device)
        d_real_labels = th.ones(pred_fake.shape, dtype=th.float32, device=self.device)

        d_real_loss = self.dis_loss(pred_real, d_real_labels)
        d_fake_loss = self.dis_loss(pred_fake, d_fake_labels)
        self.d_loss = d_real_loss + d_fake_loss
        self.d_loss.backward()
        self.dis_optim.step()

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if iter_num % 500 == 0:
            logger.add_scalars('scalar/loss', {
                'g_loss': self.g_fake_loss,
                'd_loss': self.d_loss,
                'p_loss': self.perceptual_loss
            }, iter_num)

            ra = make_grid(self.real_a, normalize=True, nrow=4)
            rb = make_grid(self.real_b, normalize=True, nrow=4)
            fb = make_grid(self.fake_b, normalize=True, nrow=4)

            logger.add_image('image/real_a', ra, iter_num)
            logger.add_image('image/real_b', rb, iter_num)
            logger.add_image('image/fake_b', fb, iter_num)

            self.gen_scheduler.step(iter_num)
            self.dis_scheduler.step(iter_num)

        if (iter_num + 1) % 5000 == 0:
            gan.save('p2p', EXP_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_len')
    args = parser.parse_args()

    # 412 iter = 1 epoch
    # 100 + 100 epochs

    device = 'cuda'
    GAN_EPOCHS = 50000
    DECREASE_LR_EPOCHS = 25000

    IMG_SIZE = 256
    BATCH_SIZE = 12
    label_len = int(args.label_len)

    EXP_NAME = 'refocus_final_' + str(label_len)
    logger = SummaryWriter('./log/' + EXP_NAME)

    LR = 2e-4

    gloss_cfg = LossConfig('MSE')
    dloss_cfg = LossConfig('MSE')

    goptim_cfg = OptimConfig('Adam', lr=LR, beta=(0.5, 0.999))
    doptim_cfg = OptimConfig('Adam', lr=LR, beta=(0.5, 0.999))

    dataset_cfg = DatasetConfig('FlowerFull', size=IMG_SIZE, num=label_len)
    loader_cfg = LoaderConfig('naive', batch_size=BATCH_SIZE, shuffle=True)

    # gen_cfg = ModelConfig('ResNetGen', input_nc=3, output_nc=3, ngf=64, n_blocks=6)
    gen_cfg = ModelConfig('UNetGen', input_nc=3, output_nc=3, num_downs=6, ngf=64,
                          use_resizeconv=True, ex_label=True, label_len=label_len)
    dis_cfg = ModelConfig('PatchDis', input_nc=6)

    gan_cfg = GanConfig(
        name='Pix2Pix', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=GAN_EPOCHS, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, device=device
    )
    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
    gan.save('p2p', EXP_NAME)
