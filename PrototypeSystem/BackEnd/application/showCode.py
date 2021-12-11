import os
import sys
import csv
import pandas as pd
import random
from flask import (Blueprint, request, jsonify, session)
from application.database.models import *
from application.verify import login_required
from flask import current_app as app
from application.split.main import codeSplit
from application.model.test_new import detection

bp = Blueprint('showCode', __name__, url_prefix='/showCode')

class ProjectTree():
    '''全路径列表构造树状层级结构
    [
        {'id': 1609400997.6737702, 'name': 'Test', 'children': [
            {'id': 57, 'name': 'test1.cpp'},
            {'id': 1609400997.6737702, 'name': 'test2.cpp'},
            {'id': 1609400997.6737702, 'name': 'dir', 'children': [
                {'id': 59, 'name': 'test1.cpp'},
                {'id': 1609400997.6737702, 'name': 'xxx.cpp'}]
            }]
        }
    ]
    '''
    def __init__(self, data_list, ids):
        self.data_list = data_list
        self.ids = ids
        self.tree = []

    def getProjectTree(self):
        '''创建项目结构树'''
        for i in range(len(self.data_list)):
            self.insertPath(self.tree, self.data_list[i], 0, self.ids[i])
        return self.tree

    def insertPath(self, root, path, depth, leafId):
        '''插入一个文件路径
        root:(list) 当前的根
        path:(list) [a, b, cpp] 一条完整的文件路径
        depth:(int)当前深度
        leafId:(int)叶子节点的ID值
        '''
        # 判断是否是叶子节点
        isLeaf = True if depth + 1 == len(path) else False
        if len(root) == 0:  # root为空列表
            if isLeaf:      # 当前节点是叶子节点
                return {
                    'index': leafId,
                    'name': path[depth]
                }
            else:
                root.append({
                    'index': random.randint(0, sys.maxsize),
                    'name': path[depth],
                    'children': []
                })
                root[0]['children'].append(self.insertPath(root[0]['children'], path, depth + 1, leafId))
        else:               # root不为空列表
            isExist = -1    # 当前节点是否已存在
            for i in range(len(root)):
                if root[i]['name'] == path[depth]:
                    isExist = i
                    break
            if isExist != -1:  # 当前节点已经存在
                self.insertPath(root[isExist]['children'], path, depth + 1, leafId)
            else:               # 当前节点不存在
                if isLeaf:      # 当前节点是叶子节点
                    root.append({
                        'index': leafId,
                        'name': path[depth]
                    })          # 当前节点不是叶子节点
                else:
                    root.append({
                        'index': random.randint(0, sys.maxsize),
                        'name': path[depth],
                        'children': []
                    })
                    root[len(root)-1]['children'].append(self.insertPath(root[len(root)-1]['children'], path, depth + 1, leafId))



@bp.route('/getProjectTree', methods=['GET', 'POST'])
@login_required
def getProjectTree():
    '''返回项目文件的树形结构'''
    if request.method == 'POST':
        projectId = request.get_json()['projectId']
        project = Project.query.filter_by(id=projectId).first()
        projectName = project.path.split('\\')[-1]
        files = File.query.filter_by(projectId=35).all()
        data = []
        ids = []
        for f in files:
            path = os.path.join(projectName, f.path)
            data.append(path.split('\\'))
            ids.append(f.id)
            print(path.split('\\'))
        tree = ProjectTree(data, ids).getProjectTree()
        print(jsonify(tree))
        return jsonify({
            'status': 200,
            'msg': tree
        })


def analyzeFile(fileId):
    '''调用切分和检测模型分析文件'''
    # 获取文件存放路径
    file = File.query.filter_by(id=fileId).first()
    projectPath = app.config['UPLOAD_FOLDER'] + file.project.path
    filePath = os.path.join(projectPath, file.path)

    # 切分结果在csv保存
    csvPath = codeSplit(filePath)
    # print("csvPath:", csvPath)

    # 模型分析结果写入csv
    detection(csvPath)

    # 分析结果写入数据库中
    df = pd.read_csv(csvPath)
    for index, row in df.iterrows():
        function = Function(fileId=fileId, start=row['Line_start'], end=row['Line_end'])
        db.session.add(function)
        db.session.flush()
        # print(function.id)
        types = row['Type'][1:-2].split(',')
        details = row['Detail'][1:-2].split(',')
        # print(types)
        # print(details)
        for i in range(len(types)):
            type = Type(functionId=function.id, detail=details[i], type=types[i])
            db.session.add(type)
        db.session.commit()

    # 文件标注已经检测
    file = File.query.filter_by(id=fileId).first()
    file.flag = 1
    db.session.commit()



@bp.route('/getFileRes', methods=['GET', 'POST'])
@login_required
def getFileRes():
    '''返回项目文件极其检测结果'''
    if request.method == 'POST':
        fileId = request.get_json()['fileId']
        file = File.query.filter_by(id=fileId).first()
        if file is None:
            return jsonify({
                'status': 400,
                'msg': '文件不存在！'
            })

        if session['userId'] != file.project.user.id:
            return jsonify({
                'status': 500,
                'msg': '非法访问！'
            })

        # 调用切分和检测模型
        if file.flag == 0:
            analyzeFile(fileId)

        error = []
        functions = Function.query.filter_by(fileId=file.id)
        for f in functions:
            types = Type.query.filter_by(functionId=f.id)
            for t in types:
                error.append({
                    'starterrorline': f.start,
                    'enderrorline': f.end,
                    'errorinfo': t.detail
                })
        return jsonify({
            'status': 200,
            'msg': {
                'filePath': os.path.join(file.project.path, file.path),
                'error': error
            }
        })