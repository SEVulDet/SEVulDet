from flask import (Blueprint, render_template, url_for, g, redirect, request, session, jsonify)
import functools
import hashlib
from application.database.models import *
from application.database.dbfunc import *

bp = Blueprint('verify', __name__, url_prefix='/verify')

def login_required(view):
    '''登录验证'''
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if session['userId'] is None:
            return jsonify({'status': 500, 'msg': 'Internal Server Error'})
            # return redirect(url_for('auth.login'))
        return view(**kwargs)

    return wrapped_view


@bp.route('/login', methods=['GET', 'POST'])
def login():
    '''用户登录处理'''
    if request.method == 'POST':
        data = request.get_json()
        msg = [
            {'status': 200, 'msg': 'OK'},
            {'status': 400, 'msg': 'Bad Request'},
            {'status': 500, 'msg': 'Internal Server Error'}
        ]
        if 'userName' not in data or 'password' not in data:    # 参数错误
            return jsonify(msg[1])
        passwordMD5 = hashlib.md5(data['password'].encode()).hexdigest()
        user = User.query.filter_by(userName=data['userName'], password=passwordMD5).first()
        if user is None:    # 账号或密码错误
            return jsonify(msg[2])
        # 创建session
        session.clear()
        session['userId'] = user.id
        session['authority'] = user.authority
        return jsonify(msg[0])



@bp.route('/logout')
@login_required
def logout():
    '''用户退出登录'''
    session.clear()
    # return redirect(url_for('index'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    '''用户注册'''
    '''用户登录处理'''
    if request.method == 'POST':
        data = request.get_json()
        msg = [
            {'status': 200, 'msg': 'OK'},
            {'status': 400, 'msg': 'Bad Request'},
            {'status': 1, 'msg': 'this user name is already exist!'},
            {'status': 2, 'msg': 'password is inconsistent!'}
        ]
        # 检测是否存在userName, password1, password2
        if 'userName' not in data or 'password1' not in data or 'password2' not in data:    # 参数错误
            return jsonify(msg[1])
        # 检测用户是否已存在
        elif User.query.filter(User.userName == data['userName']).scalar():
            return jsonify(msg[2])
        # 检测密码是否一致
        elif data['password1'] != data['password2']:
            return jsonify(msg[3])
        DbAdd.add_user(data['userName'], hashlib.md5(str(data['password1']).encode()).hexdigest())
        # 创建session
        user = User.query.filter_by(userName=data['userName']).first()
        session.clear()
        session['userId'] = user.id
        session['authority'] = user.authority
        return jsonify(msg[0])

