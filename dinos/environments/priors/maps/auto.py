from exlab.utils.io import parameter


def lazyImport(path, className):
    def importer():
        module = __import__(path, fromlist=[className])
        return getattr(module, className)
    
    return importer


class Mapper(object):
    DEFAULT = {}

    @classmethod
    def register(cls, environmentClassName, featureMapPath, featureMapClassName, dict_=None):
        dict_ = parameter(dict_, cls.DEFAULT)
        dict_[environmentClassName] = lazyImport(featureMapPath, featureMapClassName)
    
    @classmethod
    def get(cls, environmentClass, dict_=None):
        dict_ = parameter(dict_, cls.DEFAULT)
        importer = dict_.get(environmentClass.__name__)
        if not importer:
            return
        return importer()()


Mapper.register('PlaygroundEnvironment',
                'dinos.environments.priors.maps.playground', 'PlaygroundFeatureMap')
