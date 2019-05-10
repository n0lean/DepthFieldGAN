import abc
import logging
import inspect


class ConfigHelper(abc.ABC):
    accepted_type = [int, str, float, list]

    def __init__(self):
        self._base_class_ = None
        self._name_ = ''

    @property
    def base(self):
        return self._base_class_

    @base.setter
    def base(self, base_class):
        self._base_class_ = base_class

    @property
    def name(self):
        return self._name_

    @name.setter
    def name(self, name):
        self._name_ = name

    def get(self, *args, **kwargs):
        acc_args = self.base.get_argumentrs(self.name)
        arguments = {key: self.__dict__[key] for key in acc_args}
        return self.base.create(**arguments)

    def __str__(self):
        return 'Config object:\n' + '\n'.join([str(key) + ':' + str(val) for key, val in self.__dict__.items()])

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        for key in self.__dict__:
            val1 = self.__dict__[key]
            val2 = other.__dict__[key]
            if val1 != val2:
                return False
        return True


class BaseHelper(abc.ABC):
    subclasses = {}

    @classmethod
    def create(cls, **kwargs):
        name = kwargs.pop('name')
        if name not in cls.subclasses:
            logging.error('{} does not exist'.format(name))
            raise ValueError('{} does not exist'.format(name))
        return cls.subclasses[name](**kwargs)

    @classmethod
    def register(cls, name):
        def dec(subclass):
            cls.subclasses[name] = subclass
            return subclass
        return dec

    @classmethod
    def get_arguments(cls, name):
        args = inspect.getfullargspec(cls.subclasses[name]).args
        args.remove('self')
        return args
