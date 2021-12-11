<template>
    <div class="login-wrap">
        <img src="../../assets/img/bgi1.jpg" />
        <div class="ms-login">
            <div class="ms-title">漏洞挖掘系统</div>
            <el-form :model="param" :rules="rules" ref="login" label-width="0px" class="ms-content">
                <el-form-item prop="userName">
                    <el-input v-model="param.userName" placeholder="用户名">
                        <el-button slot="prepend" icon="el-icon-lx-people"></el-button>
                    </el-input>
                </el-form-item>
                <el-form-item prop="password">
                    <el-input type="密码" placeholder="password" v-model="param.password" @keyup.enter.native="submitForm()">
                        <el-button slot="prepend" icon="el-icon-lx-lock"></el-button>
                    </el-input>
                </el-form-item>
                <div class="login-btn">
                    <el-button type="primary" @click="submitForm()">登录</el-button>
                </div>
                <div class="login-btn">
                    <el-button type="success" @click="goRegister()">去注册</el-button>
                </div>
            </el-form>
        </div>
    </div>
</template>

<script>
import * as api from '@/api/user.js';
	
export default {
    data: function() {
        return {
            param: {
                userName: 'wys',
                password: '123'
            },
            rules: {
                userName: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
                password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
            }
        };
    },
    methods: {
        goRegister() {
            this.$router.push('/register');
        },
        submitForm() {
            this.$refs.login.validate(valid => {
                if (valid) {
					api.login({"userName": this.param.userName,"password": this.param.password}).then((response)=>{
							this.$message.success('登录成功');
							localStorage.setItem('ms_username', this.param.userName);
							this.$router.push('/');
						}).catch(error => console.log(error));
                } else {
                    this.$message.error('请输入账号和密码');
                    console.log('error submit!!');
                    return false;
                }
            });
        }
    }
};
</script>

<style scoped>
.login-wrap {
    position: absolute;
    width: 100%;
    height: 100%;
    /* background: rgb(64, 158, 255); */
    /* background-image: url(../../assets/img/login-bg.jpg); */
    /* background: url(../../assets/img/bgi1.jpg) no-repeat center; */
    /* background-size: 100%; */
}
img {
    display: block;
    width: 100%;
    height: 100%;
}
.ms-title {
    width: 100%;
    line-height: 50px;
    text-align: center;
    font-size: 20px;
    /* color: #fff; */
    border-bottom: 1px solid #ddd;
}
.ms-login {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 350px;
    margin: -190px 0 0 -175px;
    border-radius: 5px;
    /* color: #fff; */
    background: #fff;
    /* background: rgba(255, 255, 255, 0.3); */
    overflow: hidden;
}
.ms-content {
    padding: 30px 30px;
}
.login-btn {
    text-align: center;
}
.login-btn button {
    width: 100%;
    height: 36px;
    margin-bottom: 10px;
}
.login-tips {
    font-size: 12px;
    line-height: 30px;
    color: #fff;
}
</style>
