<template>
    <div class="wrapper">
        <v-head></v-head>
        <div class="content-box" :class="{'content-collapse':collapse}">
            <div class="content">
                <transition name="move" mode="out-in">
                    <keep-alive :include="tagsList">
                        <router-view></router-view>
                    </keep-alive>
                </transition>
                <el-backtop target=".content"></el-backtop>
            </div>
        </div>
    </div>
</template>

<script>
import vHead from './Header.vue';
import vTags from './Tags.vue';
import bus from './bus';
export default {
    data() {
        return {
            tagsList: [],
            collapse: false
        };
    },
    components: {
        vHead,
        vTags
    },
    created() {
        bus.$on('collapse-content', msg => {
            this.collapse = msg;
        });
    }
};
</script>

<style scoped>
.content{
    overflow: hidden;
}
</style>
