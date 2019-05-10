import torch as th
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load
from torch.utils.data import DataLoader


class LoaderConfig(ConfigHelper):
    def __init__(self, name, batch_size, shuffle, num_workers=0):
        super(LoaderConfig, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get(self, dataset):
        acc_args = BaseLoader.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        args['dataset'] = dataset
        return BaseLoader.create(**args)
        

class BaseLoader(BaseHelper):
    subclasses = {}


class LoaderSchema(Schema):
    name = fields.Str()
    batch_size = fields.Int()
    shuffle = fields.Bool()
    __model__ = LoaderConfig

    @post_load
    def make_object(self, data):
        return self.__model__(
            name=data['name'],
            batch_size=data['batch_size'],
            shuffle=data['shuffle']
        )


@BaseLoader.register('naive')
class NaiveLoader(BaseLoader, DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        super(NaiveLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
