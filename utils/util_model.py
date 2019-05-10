import torch as th
import abc
import numpy as np
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load
from utils.optim import OptimSchema
from utils.loss import LossSchema
from utils.dataloader import LoaderSchema
from dataset.config import DatasetSchema
from torch import nn
import logging
import tqdm
from torch.nn import functional as F
import math


class UtilityModelConfig(ConfigHelper):
    def __init__(self, name, **kwargs):
        super(UtilityModelConfig, self).__init__()
        self.name = name
        self.__dict__.update(kwargs)

    def get(self, load_from=None):
        acc_args = BaseUtilityModel.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        model = BaseUtilityModel.create(**args)
        if load_from is not None:
            sd = th.load(load_from)
            model.load_state_dict(sd)
            return model

        if self.scratch:
            print('Training a NaiveClassifier from scratch.')
            dataset = self.train_data_cfg.get()
            loader = self.dataloader_cfg.get(dataset)
            model = NaiveClassifier(self.img_sz)
            optim = th.optim.Adam(model.parameters())
            cri = th.nn.NLLLoss()
            model.train()
            model.to(self.device)
            for epoch in tqdm.tqdm(range(self.epochs)):
                totloss = 0.0
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(self.device).float(), target.to(self.device)
                    optim.zero_grad()
                    output = model(data)
                    loss = cri(output, target)
                    loss.backward()
                    optim.step()
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, totloss))
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, totloss))
            state_dict = model.state_dict()
            th.save(state_dict, self.weight_pth)
            print('Weight saved to {}'.format(self.weight_pth))

            test_data = self.test_data_cfg.get()
            test_loader = self.testloader_cfg.get(test_data)
            model.eval()
            test_loss = 0
            correct = 0
            with th.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device).float(), target.to(self.device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target).item()  # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        else:
            if self.scratch:
                logging.warning('Weight file exists.')
                print('Weight file exists.')
            state_dict = th.load(self.weight_pth)
            model.load_state_dict(state_dict)
            print('Loading model from {}'.format(self.weight_pth))
        return model


class UtilityModelSchema(Schema):
    name = fields.Str()
    scratch = fields.Str()
    weight_pth = fields.Str()
    train_data_cfg = fields.Nested(DatasetSchema)


class BaseUtilityModel(BaseHelper):
    subclasses = {}


@BaseUtilityModel.register('NaiveClassifier')
class NaiveClassifier(BaseUtilityModel, nn.Module):
    def __init__(self, img_sz):
        super(NaiveClassifier, self).__init__()
        self.img_sz = img_sz
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if img_sz == 32:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        else:
            self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x, feature=False):
        # embed()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.img_sz == 32:
            x = x.view(-1, 16 * 5 * 5)
        else:
            x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if feature:
            return x
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = th.nn.Parameter(th.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


@BaseUtilityModel.register('sphere')
class sphere20a(nn.Module, BaseUtilityModel):
    def __init__(self, classnum=10574, feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        # input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)
        self.fc6 = AngleLinear(512, self.classnum)

        self.g_avg = nn.AvgPool2d(4)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        if self.feature:
            x = self.g_avg(x)
            return x.view(x.size(0), -1)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        # if self.feature:
        #     return x

        x = self.fc6(x)
        return x