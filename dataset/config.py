from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load


class DatasetConfig(ConfigHelper):
    def __init__(self, name, **kwargs):
        super(DatasetConfig, self).__init__()
        self.name = name
        self.__dict__.update(kwargs)

    def get(self):
        acc_args = BaseDataset.get_arguments(self.name)
        args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
        args['name'] = self.name
        return BaseDataset.create(**args)


class BaseDataset(BaseHelper):
    subclasses = {}


class DatasetSchema(Schema):
    name = fields.Str()
    mode = fields.Int()
    sig = fields.Float()
    num_per_mode = fields.Int()
    max_val = fields.Int()
    stack = fields.Bool()
    train = fields.Bool()
    along_width = fields.Bool()

    __model__ = DatasetConfig

    @post_load
    def make_object(self, data):
        return self.__model__(
            name=data['name'],
            mode=data['mode'],
            sig=data['sig'],
            num_per_mode=data['num_per_mode'],
            max_val=data['max_val']
        )
