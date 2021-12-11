from flask import (Blueprint, render_template)
from application.database.models import *


bp = Blueprint('test', __name__, url_prefix='/test')

@bp.route('/')
def index():
    # return "hello"
    users = User.query.all()
    return render_template('test.html', users=users)