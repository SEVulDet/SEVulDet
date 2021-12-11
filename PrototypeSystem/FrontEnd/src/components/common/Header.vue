<template>
    <div class="header">
        <div class="header-left">
			
			<!--<div class="syslogo">
			    <img src="../../assets/img/img.jpg" />
			</div>
			:default-active="defaultActive"
			default-active="this.$route.path"
			-->
			<img src="../../assets/img/bug fix.png" alt="logo" style="height: 40px; margin-left: 30px;">
            <el-menu
                
				:default-active="defaultActive"
                background-color="#242f42"
                text-color="white"
				
                class="el-menu-demo"
                mode="horizontal"
                active-text-color="#ffd04b"
				
                router>
                <template v-for="item in items">
                    <el-menu-item :index="item.index" :key="item.index" trigger="click" @click="changeContent(item.name)">{{ item.name }}</el-menu-item>
                </template>
            </el-menu>
        </div>
        <div class="header-right">
            <div class="header-user-con">
                <!-- 全屏显示 -->
                <div class="btn-fullscreen" @click="handleFullScreen">
                    <el-tooltip effect="dark" :content="fullscreen?`取消全屏`:`全屏`" placement="bottom">
                        <i class="el-icon-rank"></i>
                    </el-tooltip>
                </div>
                <!-- 用户头像 -->
                <div class="user-avator">
                    <img src="../../assets/img/img.jpg" />
                </div>
                <!-- 用户名下拉菜单 -->
                <el-dropdown class="user-name" trigger="click" @command="handleCommand">
                    <span class="el-dropdown-link">
                        {{username}}
                        <i class="el-icon-caret-bottom"></i>
                    </span>
                    <el-dropdown-menu slot="dropdown">
                        <el-dropdown-item divided command="loginout">退出登录</el-dropdown-item>
                    </el-dropdown-menu>
                </el-dropdown>
            </div>
        </div>
    </div>
</template>
<script>
import bus from '../common/bus';
export default {
    data() {
        return {
            collapse: false,
            fullscreen: false,
            name: 'linxin',
            message: 2,
            defaultActive: '1',
            items: [
                {
                    index: '1',
                    name: '系统首页'
                },{
                    index: '2',
                    name: '文件上传'
                },{
                    index: '3',
                    name: '漏洞查看'
                },
            ]
        };
    },
    computed: {
        username() {
            let username = localStorage.getItem('ms_username');
            return username ? username : this.name;
        }
    },
    methods: {
        changeContent(name){
            if(name=="文件上传"){
                this.$router.push('/upload')
            }else if(name=="漏洞查看"){
                this.$router.push('/codeallshow')
            }else if(name=="系统首页"){
                this.$router.push('/dashboard')
            }
        },
        // 用户名下拉菜单选择事件
        handleCommand(command) {
            if (command == 'loginout') {
                localStorage.removeItem('ms_username');
                this.$router.push('/login');
            }
        },
        // 侧边栏折叠
        collapseChage() {
            this.collapse = !this.collapse;
            bus.$emit('collapse', this.collapse);
        },
        // 全屏事件
        handleFullScreen() {
            let element = document.documentElement;
            if (this.fullscreen) {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitCancelFullScreen) {
                    document.webkitCancelFullScreen();
                } else if (document.mozCancelFullScreen) {
                    document.mozCancelFullScreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            } else {
                if (element.requestFullscreen) {
                    element.requestFullscreen();
                } else if (element.webkitRequestFullScreen) {
                    element.webkitRequestFullScreen();
                } else if (element.mozRequestFullScreen) {
                    element.mozRequestFullScreen();
                } else if (element.msRequestFullscreen) {
                    // IE11
                    element.msRequestFullscreen();
                }
            }
            this.fullscreen = !this.fullscreen;
        }
    },
    mounted() {
        if (document.body.clientWidth < 1500) {
            this.collapseChage();
        }
    }
};
</script>
<style scoped>
.el-menu.el-menu--horizontal{
  border: none;
  height: 100%;
}
.el-menu--horizontal>.el-menu-item{
    height: 70px;
    line-height: 70px;
}
 .el-menu-item{/*导航栏字体*/
	  font-size:16px;
  }
.header {
    position: relative;
    box-sizing: border-box;
    width: 100%;
    height: 70px;
    font-size: 22px;
    color: #fff;
    background-color: #242f42;
}
.collapse-btn {
    float: left;
    padding: 0 21px;
    cursor: pointer;
    line-height: 70px;
}
.header .logo {
    float: left;
    width: 250px;
    line-height: 70px;
}
.header-right {
    float: right;
    padding-right: 50px;
}
.header-left{
    float: left;
    padding-right: 50px;
    height: 100%;
    border: 0;
	/*以下两为更改*/
	display: flex;
	align-items: center;
}
.header-user-con {
    display: flex;
    height: 70px;
    align-items: center;
}
.btn-fullscreen {
    transform: rotate(45deg);
    margin-right: 5px;
    font-size: 24px;
}
.btn-bell,
.btn-fullscreen {
    position: relative;
    width: 30px;
    height: 30px;
    text-align: center;
    border-radius: 15px;
    cursor: pointer;
}
.btn-bell-badge {
    position: absolute;
    right: 0;
    top: -2px;
    width: 8px;
    height: 8px;
    border-radius: 4px;
    background: #f56c6c;
    color: #fff;
}
.btn-bell .el-icon-bell {
    color: #fff;
}
.user-name {
    margin-left: 10px;
}
.user-avator {
    margin-left: 20px;
}
.user-avator img {
    display: block;
    width: 40px;
    height: 40px;
    border-radius: 50%;
}
.el-dropdown-link {
    color: #fff;
    cursor: pointer;
}
.el-dropdown-menu__item {
    text-align: center;
}

.header-left img{
	display: block;
	width: 40px;
	height: 40px;
	/*border-radius: 50%;
	*/
}
</style>
