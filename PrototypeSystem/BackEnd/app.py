# app.py
from flask_migrate import Migrate

from application import init_app

app = init_app("dev")

# 启用数据迁移工具
# Migrate(app, db)


if __name__ == "__main__":
    app.run()




