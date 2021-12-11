import time

from application import db
from application.database.models import *

def init_db():
    '''创建关系表'''
    db.create_all()

class DbQuery:
    # constructor
    def __init__(self):
        pass

    @staticmethod
    def query_model_record(modelName, other_filters=[]):
        '''查找模型中符合条件的第一条记录'''
        model = MODELS[modelName]
        sql = model.query
        # other conditions
        for i in range(len(other_filters)):
            sql = sql.filter(other_filters[i])
        result = sql.first()
        return result

    @staticmethod
    def query_model_records(modelName, other_filters=[]):
        '''查找模型中符合条件的记录'''
        model = MODELS[modelName]
        sql = model.query
        # other conditions
        for i in range(len(other_filters)):
            sql = sql.filter(other_filters[i])
        result = sql.all()
        return result

    @staticmethod
    def query_lastest_commit_record(modelName, filters=[]):
        '''查找模型符合条件的最新记录'''
        sql = MODELS[modelName].query
        # other conditions
        for i in range(len(filters)):
            sql = sql.filter(filters[i])
        sql = sql.order_by(Project.commitTime.desc())
        result = sql.first()
        return result

    


class DbAdd():
    # constructor
    def __init__(self):
        pass

    @staticmethod
    def add(modelName, condiction=[]):
        '''给user表增加记录'''
        model = MODELS['modelName']
        pass

    @staticmethod
    def add_user(userName, password, authority=0):
        '''给user表增加记录'''
        user = User(userName=userName, password=password, authority=authority)
        db.session.add(user)
        db.session.commit()


class DbDelete():
    # constructor
    def __init__(self):
        pass

    @staticmethod
    def delete(modelName):
        '''给指定表删除记录'''
        pass

class DbInsert():
    # constructor
    def __init__(self):
        pass

    @staticmethod
    def insert_project(userId, path, commitTime):
        '''插入一条project记录'''
        project = Project(userId=userId, path=path, commitTime=commitTime)
        db.session.add(project)
        db.session.commit()

    @staticmethod
    def insert_file(projectId, path):
        '''插入一条file记录'''
        file = File(projectId=projectId, path=path)
        db.session.add(file)
        db.session.commit()
