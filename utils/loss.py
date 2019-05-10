import torch as th
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load


class LossConfig(ConfigHelper):
    def __init__(self, name):
        super(LossConfig, self).__init__()
        self.name = name

    def get(self):
        acc_args = BaseLoss.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        return BaseLoss.create(**args)


class BaseLoss(BaseHelper):
    subclasses = {}


class LossSchema(Schema):
    name = fields.Str()
    __model__ = LossConfig

    @post_load
    def make_object(self, data):
        return self.__model__(name=data['name'])


@BaseLoss.register('BCE')
class BCE(BaseLoss, th.nn.BCELoss):
    pass


@BaseLoss.register('MSE')
class MSE(BaseLoss, th.nn.MSELoss):
    pass


@BaseLoss.register('L1')
class L1(BaseLoss, th.nn.L1Loss):
    pass


@BaseLoss.register('BCEWithLogits')
class BCEWithLogits(BaseLoss, th.nn.BCEWithLogitsLoss):
    pass