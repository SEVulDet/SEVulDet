import os
import time
import datetime
import zipfile
import rarfile
from flask import (Blueprint, request, session, jsonify)
from application.verify import login_required
from flask import current_app as app
from application.database.models import *
from application.database.dbfunc import *

bp = Blueprint('upload', __name__, url_prefix='/upload')

ALLOWED_UPLOAD_FILE_TYPES = ['zip', 'rar']
ALLOWED_CHECK_LANGUAGE_TYPES = ['cpp']

def allowed_upload_type(filename):
    '''允许接受的文件类型'''
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_UPLOAD_FILE_TYPES


def allowed_check_language_type(filename):
    '''允许检查的编程语言文件类型'''
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_CHECK_LANGUAGE_TYPES


@bp.route('/upload_package', methods=['GET', 'POST'])
@login_required
def upload_package():
    '''上传待测项目'''
    if request.method == 'POST':
        # 获取文件对象
        package = request.files['package']
        # 检查文件类型
        if not allowed_upload_type(package.filename):
            return jsonify({
                'status': '1',
                'msg': 'type of file is not standard'
            })
        # 判断用户文件夹是否存在
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(session['userId']))
        if not os.path.exists(user_dir):
            os.mkdir(user_dir)

        # 保存到指定文件夹
        type = package.filename.split('.')[-1]
        rand_dir = str(int(time.time()))   #随机生成目录保存
        newFileName = package.filename.split('.')[0] + rand_dir + '.' + type
        file_path = os.path.join(user_dir, newFileName)
        package.save(file_path)

        # 解压到用户目录下面
        if (type == 'zip'):
            un_zip(user_dir, newFileName, rand_dir)
        elif (type == 'rar'):
            # un_rar()
            pass
        # 项目文件记录数据库
        archive(os.path.join(user_dir, rand_dir))
        # 返回project最新的项目id
        project_id = Project.query.filter(Project.userId==session['userId'])\
            .order_by(Project.commitTime.desc()).first().id
        print(project_id)
        return jsonify({
                'status': 0,
                'msg': project_id
            })




def un_zip(file_dir, fileName, t):
    """unzip zip file
       file_dir: (str)目录路径
       fileName: (str)文件名
       t: (str)timestamp 取整字符串
    """

    file_path = os.path.join(file_dir, fileName)
    zip_file = zipfile.ZipFile(file_path)
    upzip_dir = os.path.join(file_dir, t)

    if os.path.isdir(upzip_dir):
        pass
    else:
        os.mkdir(upzip_dir)
    for names in zip_file.namelist():
        zip_file.extract(names, upzip_dir)
    zip_file.close()

def un_rar(file_name):
    """unrar zip file"""
    rar = rarfile.RarFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    # os.chdir(file_name + "_files"):
    # rar.extractall()
    # rar.close()
    
def archive(base_path):
    '''项目文件写入数据库
        user_dir: (str)用户目录路径
        commitTime: (timestamp)
    '''
    # 本次上传系统创建保存目录路径
    upload_path = app.config['UPLOAD_FOLDER']
    # 获取base_path下所有的项目名，并且遍历
    dir_list = os.listdir(base_path)
    for dir in dir_list:
        dir_path = os.path.join(base_path, dir)
        project_path = dir_path.split(upload_path)[-1]
        print(project_path)
        # 向数据库插入project记录
        t = datetime.now()
        DbInsert.insert_project(session['userId'], project_path, t)
        # # 获取刚刚插入的project.id
        other_filters = [Project.userId==session['userId'],
                         Project.path==project_path]
        project_id = DbQuery.query_model_record('project', other_filters).id
        for root, ds, fs in os.walk(dir_path):
            cur_path = root.split(dir_path)[-1][1:]
            # print(file_path)
            for f in fs:
                # 过滤不检查的编程语言文件
                if allowed_check_language_type(f):
                    file_path = os.path.join(cur_path, f)
                    # print(file_path)
                    # 向数据库插入file记录
                    DbInsert.insert_file(project_id, file_path)

@bp.route('/get_project_list', methods=['GET', 'POST'])
@login_required
def get_project_list():
    if request.method == 'POST':
        # 获取项目内容
        projects = Project.query.filter_by(userId=session['userId']).order_by(Project.commitTime.desc()).all()
        # print("projects:", projects)
        res = []
        for p in projects:
            info = {}
            info['id'] = p.id
            info['name'] = p.path.split('\\')[-1]
            info['commitTime'] = p.commitTime
            res.append(info)
        return jsonify(res)