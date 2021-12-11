'''产品上线配置文件'''
from . import Config

class ProductionConfig(Config):
    '''生产模式下的配置'''
    DEBUG = False