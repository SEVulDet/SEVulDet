<template>
    <div>
        <div class="top">
            <el-button @click="drawer = true" type="warning" plain style="margin-left: 40px;">
                点击查看使用手册
            </el-button>
            <el-drawer
                :visible.sync="drawer"
                direction="ltr">
                <div style="height: 300px; margin-left: 60px;">
                    <el-steps direction="vertical" space="300px" active="4">
                        <el-step title="使用需知" description="本系统仅支持上传zip,rar文件"></el-step>
                        <el-step title="第一步：选择文件" description="用户每次只能上传一个文件，如果想要上传其他文件，请先将当前文件删除，然后重新点击选择文件按钮"></el-step>
                        <el-step title="第二步：提交检测" description="用户点击提交检测即可将文件上传到服务器进行检测，检测过程较长，请用户耐心等待"></el-step>
                        <el-step title="第三步：检测完成" description="当检测完成时，新的一条检测记录会添加在页面右侧的用户检测记录中，点击相应的检测记录，即可查看相应的检测结果"></el-step>
                    </el-steps>
                </div>
            </el-drawer>
            <el-row :gutter="12">
                <el-col :span="8">
                    <el-card class="box-card" shadow="always" style="margin: 40px 0 20px 40px;">
                        <el-upload
                            class="upload-demo"
                            ref="upload"
                            :limit="1"
                            accept=".zip,.rar"
                            action="api/upload/upload_package"
                            :http-request="handleUpload"
                            :on-remove="handleRemove"
                            :on-change="handleChange"
                            :file-list="fileList"
                            :auto-upload="false">
                            <el-button class="btn-upload" slot="trigger" size="medium" type="primary" icon="el-icon-search">选取文件</el-button>
                            <el-button style="margin: 10px;" size="medium" type="success" @click="submitUpload" icon="el-icon-upload">提交检测</el-button>
                        </el-upload>
                        <el-steps :space="200" :active="status" finish-status="success" process-status="wait" :align-center=true>
                            <el-step title="已选择文件"></el-step>
                            <el-step title="正在检测"></el-step>
                            <el-step title="完成"></el-step>
                        </el-steps>
                    </el-card>
                </el-col>
                <el-col :span="15" shadow="always" style="margin-left: 50px;">
                    <el-card class="box-card timetamp" shadow="always" >
                        <el-timeline :reverse=true>

                            <el-timeline-item v-on:click.native="lookUpResult(val.id)" v-for="val in records" :key="val.commitTime" :timestamp="val.commitTime" placement="top">
                                <el-card shadow="never" style="cursor: pointer;">
                                    <h4>{{val.name}}</h4>
                                    <p>你 提交于 {{val.commitTime}}</p>
                                </el-card>
                            </el-timeline-item>

                        </el-timeline>
                    </el-card>
                </el-col>
            </el-row>
            
        </div>



        <div class="foot">
            <div class="revelent">
                <div class="cl-1 cl ">
                    <div class="item0">
                        联系我们
                    </div>
                    <div class="item1">
                        问卷调研客服：123456
                    </div>
                    <div class="item1">
                        问卷调研QQ交流群：123456
                    </div>
                    <div class="item1">
                        反馈问题：123456@qq.com
                    </div>
                </div>
                <div class="cl-2 cl ">
                    <div class="item0">
                        友情链接
                    </div>
                    <div class="item2">
                        <a href="http://" target="_blank" rel="noopener noreferrer" style="color:#788895">开发者中心</a>
                    </div>
                    <div class="item2">
                        <a href="http://" target="_blank" rel="noopener noreferrer" style="color:#788895">开发团队</a>
                    </div>
                </div>
                <div class="cl-3 cl ">
                    <div class="item0">
                        关注微信公众号
                    </div>
                </div>
            </div>
            <div class="message">
                网络安全课程作业
            </div>
        </div>

    </div>
</template>

<script>
// import axios from 'axios';
import * as api from '@/api/upload.js';

export default {
    data() {
        return {
            records:[
                
            ],
            status:0,
            drawer: false,
            fileList: [
                
            ]
        };
    },
    methods: {
        submitUpload() {
            this.$refs.upload.submit();
        },
        handleRemove(file, fileList) {
            this.fileList = []
            this.status = 0
        },
        handleChange(file,fileList) {
            if (fileList.length > 0) {
                this.fileList = [fileList[fileList.length - 1]] 
                this.status = 1
            }
        },
        handleUpload(params) {
            const _file = params.file;
            
            var formdata = new FormData();
            formdata.append("package", _file);
            
            api.postFile(formdata).then((response)=>{
                if(response.status==200){
                    this.status = 2;
                }
            }).catch(error =>{
                console.log("error: ", error)
            });
		  
        },
        lookUpResult(projectid) {
            console.log("应该跳转到相应的页面");
            this.$router.push({name:'codeallshow',params:{projectId:projectid}});
        }
    },
    created() {
        api.getRecords().then((response)=>{
            //console.log(response.data)
            this.records = response.data
        })
    }
}
</script>

<style scoped>
.container{
    height: 460px;
    overflow: hidden;
}
.el-row{
    padding:50 0;
}
.el-button{
    padding:10px 58px;
}
.el-col{
    margin: 10px 0;
}
.timetamp{
    max-height: 450px; 
    overflow:scroll;
}
.timetamp::-webkit-scrollbar {
  width : 10px; 
  height: 1px;
}
.el-steps{
    padding-top: 50px;
}
.foot{
    width: 100%;
    height: 300px;
    position: absolute;
    bottom: 0;
    right: 0;
}
.cl{
    font-family: "microsoft yahei",Helvetica,Arial,sans-serif;
    font-size: 15px;
    line-height: 1.42857143;
    color: #afc8de;;
    height: auto;
    display: block;
    float: left;
    padding-top: 3%;
}
.cl-1{
    width: 30%;
    padding-left: 16%;
    overflow: hidden;
}
.item1, .item2{
    font-size: 12px;
    color: #788895;
    display: block;
    margin: 8px 0;
    text-decoration: none;
}
.cl-2{
    width: 30%;
    padding-left: 30px;
}
.cl-2>a{
    color: #788895;
}
.cl-3{
    width: 20%;
}
.revelent{
    bottom: 0;
    width: 100%;
    background-color: #2c3d4c;
    height: 225px;
}
.message{
    width: 100%;
    height: 75px;
    text-align: center;
    background-color: #293947;
    color: #788895;
    font-style: 14px;
    line-height: 70px;
}
</style>