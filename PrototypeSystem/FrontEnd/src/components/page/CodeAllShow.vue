<template>

	<el-container>
	  <el-aside  width="auto">
		<el-menu>
		 <el-menu-item>
		        <template slot="title"><i class="el-icon-document-copy"></i>漏洞代码</template>
		</el-menu-item>
		<menutree :data="menu_data" @codeinfochange="infochange($event)"></menutree>
		</el-menu>
	  </el-aside>
	  <el-container>
	    <el-main>
		  <CodeView ref="codeshow" :value="MainCode"></CodeView>
	    </el-main>
		<el-footer height="auto">
			<!-- 拖动功能 -->
			<div
			          class="resize"
			          @mousedown="mouseDownHand($event,'footer')"
			        >
			</div>
			<el-table :data="error"
					  id="footer"
					  border
					  :header-cell-style="{backgroundColor:'#F0F8FF'}"
					  @row-click="showerror">
			        <el-table-column prop="starterrorline" label="错误行数" width="180"
					 >
			        </el-table-column>
			        <el-table-column prop="errorinfo" label="错误信息" >
			        </el-table-column>
			</el-table>
		</el-footer>
	  </el-container>
	</el-container>
</template>

<script>
  import CodeView from '@/components/page/Codeview';
  import menutree from "@/components/page/CodeList";
  import dedent from 'dedent';
  import * as api from '@/api/codeshow.js';
	
  export default {
	name: "allcodeshow",
	components: {
	  CodeView,
	  menutree,
	},
    data() {
      return {
		MainCode: dedent`这里将展示用户上传的代码的漏洞信息
		`  ,
		menu_data: [],
		projectId: null,
		error:[],
		
      }
    },
	created(){
		this.projectId=this.$route.params.projectId;
		api.getlist({projectId:this.projectId}).then((response)=>{
		        this.menu_data=response.data.msg;
		      }).catch(error => console.log(error));
	},
	methods: {
		updatemenu(pid){
			api.getlist({projectId:pid}).then((response)=>{
				let menu=this.menu_data.slice(0);
			    menu=response.data.msg;
				this.menu_data=menu;
			    }).catch(error => console.log(error));
		},
		showerror(row){
			this.$refs.codeshow.lighterBg(row.starterrorline,row.enderrorline);	
		},
		mouseDownHand(e,eleid) {
			let init = e.clientY;
			let parent = document.getElementById(eleid);
			let initHeight = parent.offsetHeight;
			document.onmousemove = function(e) {
			    let end = e.clientY;
			    let newHeight = init - end + initHeight;
			    parent.style.height = newHeight + "px";
			};
			document.onmouseup = function() {
				document.onmousemove = document.onmouseup = null;
		    };
		},
		infochange(info){
			// console.log(info.filepath);
			let pathfile = "/static/uploads/" + info.filepath;
			api.getfilecode(pathfile).then((response)=>{
			        this.MainCode = response.data;
					this.error = info.error;
			      }).catch(error => console.log(error));
		},
		
	}
};
</script>

<style>
  /* .el-header {
    background-color: #B3C0D1;
    color: #333;
    line-height: 60px;
	font-size: 20px;
  } */
  
  .el-container{
	  width: 100%;
	  height: 100%;
	  background-color: #ffffff;
  }
  
  .el-aside {
    color: #333;
	background-color: #ffffff;
  }
  
  .el-footer{
	  padding-left: 0px;
  }

  .el-main{
	  padding: 0px;
	  padding-left: 0px;
	  padding-bottom: 30px;
	  height: 800px;
  }
  
  .el-menu{
	  background-color: #ffffff;
	  width:auto;
  }
  
  .el-menu-item{
	  font-size:22px;
  }
  
  .el-submenu__title{
	  font-size: 20px;
  }
  
  .el-table{
	  font-size: 20px;
  }
  
  .el-table th {
	  background-color: #ffffff;
  }
  .el-table tr {
	  background-color: #ffffff;
  }
  
    /*拖拽区div样式*/
  .resize {
	  cursor: ns-resize;
     /* position: relative; */
      background-color: #ffffff;
      /* border-radius: 5px; */
      margin-top: 10px;
      width: auto;
      height: 10px;
      /* background-size: cover; */
      background-position: center;
      z-index: 99999;
      font-size: 32px;
      color: white;
    }
    /*拖拽区鼠标悬停样式*/
/*  .resize:hover {
     color: #aa0000;
    } */

</style>

