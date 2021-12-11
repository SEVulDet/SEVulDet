<template>
<div class="menutree">
    <label v-for="(menu, index) in data" :key="menu.index">
    <el-submenu :index="menu.index + ''" v-if="menu.children">
        <template slot="title">
		<i class="el-icon-folder"></i>
        <span>{{menu.name}}</span>
        </template>
        <label>
        <menutree :data="menu.children" @codeinfochange="seekchangecode($event)"></menutree>
        </label>
    </el-submenu>
    <el-menu-item v-else :index="menu.index + ''" @click="changecode(menu.index)">
        <span slot="title">{{menu.name}}</span>
    </el-menu-item>
    </label>
</div>
</template>

<script>
  import dedent from 'dedent';
	import * as api from '@/api/codeshow.js';
	
  export default {
	name: "menutree",
	data() {
		return {
			menu_data: {},
			filepath:'',
			error:[],
		};
	},
	props: {
		data: Array
	},
	methods:{
		changecode(id){
			api.getfile({"fileId":id}).then((response)=>{
				this.filepath = response.data.msg.filePath;
				this.error = response.data.msg.error;
				let info = {
					filepath: this.filepath,
					error: this.error,
				};
				this.$emit("codeinfochange",info);
			}).catch(error => console.log(error));
		},
		seekchangecode(info){
			this.$emit("codeinfochange",info);
		}
	}
};
</script>

<style>
</style>
