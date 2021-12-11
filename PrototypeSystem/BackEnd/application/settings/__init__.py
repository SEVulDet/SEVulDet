'''项目配置文件'''

class Config(object):
    '''项目配置核心类'''
    # 调试模式
    DEBUG = True

    # mysql数据库的配置信息
    # 'SQLALCHEMY_DATABASE_URI ="数据库类型://用户名:密码@ip:port:库名?指定字符集编码"'
    # mysql: // root: 123456 @ localhost / nsc utf8mb4 utf8mb4_general_ci
    SQLALCHEMY_DATABASE_URI = "mysql://root:123456@localhost/nsc?charset=utf8mb4"
    # 动态追踪修改设置，如未设置只会提示警告
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 查询时会显示原始SQL语句
    SQLALCHEMY_ECHO = False
    # secret key
    SECRET_KEY = 'TPmi4aLWRbyVq8zu9v82dWYW1'
    # 数据存放路径
    UPLOAD_FOLDER = r'application\static\uploads'
    # 上传文件大小限制
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024