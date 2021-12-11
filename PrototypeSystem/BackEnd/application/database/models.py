# coding=utf-8
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


# 创建关系表,不再创建模型,一般用于表与表之间的多对多场景
# """
# 表关系变量 = db.Table(
#     "关系表表名",
#     db.Column('字段名', 字段类型, 字段选项),  # 普通字段
#     db.Column("字段名", 字段类型, db.ForeignKey("表名.id")),
#     db.Column("字段名", 字段类型, db.ForeignKey("表名.id")),
# )
# """

db = SQLAlchemy()

class User(db.Model):
    '''用户信息表'''
    __tablename__ = "user"
    id = db.Column(db.Integer, autoincrement=True, primary_key=True, comment="用户ID")
    userName = db.Column(db.String(25), unique=True, index=True, comment="用户名")
    password = db.Column(db.String(25), nullable=True, comment="密码")
    authority = db.Column(db.Integer, default=0, comment="权限")
    projects = db.relationship('Project', backref='user', lazy=True)

class Project(db.Model):
    __tablename__ = "project"
    id = db.Column(db.Integer, autoincrement=True, primary_key=True, comment="待测项目ID")
    userId = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, comment="上传者ID")
    path = db.Column(db.String(255), nullable=True, comment="待测项目路径")
    commitTime = db.Column(db.DateTime, default=datetime.now, comment="上传时间")
    files = db.relationship('File', backref='project', lazy=True)

class File(db.Model):
    __tablename__ = "file"
    id = db.Column(db.Integer, autoincrement=True, primary_key=True, comment="待测文件ID")
    projectId = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True, comment="所属文件ID")
    path = db.Column(db.String(255), nullable=True, comment="待测文件路径")
    flag = db.Column(db.Integer, default=0, comment="是否被检测")
    funcitons = db.relationship('Function', backref='file', lazy=True)

class Function(db.Model):
    __tablename__ = "function"
    id = db.Column(db.Integer, autoincrement=True, primary_key=True, comment="待测文件ID")
    fileId = db.Column(db.Integer, db.ForeignKey('file.id'), nullable=True, comment="所属函数ID")
    end = db.Column(db.Integer, comment="函数起始位置")
    start = db.Column(db.Integer, comment="函数终止位置")
    types = db.relationship('Type', backref='function', lazy=True)

class Type(db.Model):
    __tablename__ = "type"
    id = db.Column(db.Integer, autoincrement=True, primary_key=True, comment="待测文件ID")
    functionId = db.Column(db.Integer, db.ForeignKey('function.id'), nullable=True, comment="所属函数ID")
    type = db.Column(db.String(255), comment="错误类型")
    detail = db.Column(db.String(255), comment="详细信息")

MODELS = {
    'user': User,
    'project': Project,
    'file': File,
    'function': Function,
    'type': Type
}