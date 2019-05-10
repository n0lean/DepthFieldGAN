import torch as th
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load


class OptimConfig(ConfigHelper):
    def __init__(self, name, lr, beta=(0.9, 0.5)):
        super(OptimConfig, self).__init__()
        self.name = name
        self.lr = lr
        if beta is not None:
            assert isinstance(beta, tuple) or isinstance(beta, list)
            assert len(beta) == 2
        self.beta = [*beta]

    def get(self, parameters):
        acc_args = BaseOptim.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        args['params'] = parameters
        return BaseOptim.create(**args)


class BaseOptim(BaseHelper):
    subclasses = {}


class OptimSchema(Schema):
    name = fields.Str()
    lr = fields.Float()
    beta = fields.List(fields.Float())
    __model__ = OptimConfig

    @post_load
    def make_object(self, data):
        return self.__model__(
            name=data['name'],
            lr=data['lr'],
            beta=data['beta']
        )


@BaseOptim.register('Adam')
class Adam(th.optim.Adam, BaseOptim):
    pass


@BaseOptim.register('RMSprop')
class Adam(th.optim.RMSprop, BaseOptim):
    pass
