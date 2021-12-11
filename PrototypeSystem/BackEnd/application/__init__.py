import os
from flask import Flask

from application.settings.dev import DevelopmentConfig
from application.settings.prob import ProductionConfig
from application.database.models import *



config = {
    "dev": DevelopmentConfig,
    "prob": ProductionConfig
}


def init_app(config_name):

    '''项目的初始化函数'''
    app = Flask(__name__)

    # 设置配置类
    Config = config[config_name]

    # 指定数据存放的具体路径
    Config.UPLOAD_FOLDER = os.path.join(os.getcwd(), Config.UPLOAD_FOLDER)

    # 加载配置
    app.config.from_object(Config)

    # 配置数据库连接
    db.init_app(app)
    db.create_all(app=app)

    # registering components
    from . import verify, upload, showCode, test
    app.register_blueprint(test.bp)
    app.register_blueprint(verify.bp)
    app.register_blueprint(upload.bp)
    app.register_blueprint(showCode.bp)


    return app