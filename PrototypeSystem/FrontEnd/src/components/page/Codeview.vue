<template>
		<codemirror
          ref="codeMirros"
          :value="value"
          :options="codeMirrosOptions"
		  >
		</codemirror>
</template>

<script>
  import dedent from 'dedent'
  import { codemirror } from 'vue-codemirror'
  //base style
  import 'codemirror/lib/codemirror.css'
  // import 'codemirror/theme/solarized.css'
  require("codemirror/mode/clike/clike.js")
  // import'codemirror/addon/selection/active-line.js'
  let WRAP_CLASS = "CodeMirror-activeline";
  let BACK_CLASS = "CodeMirror-activeline-background";
  let GUTT_CLASS = "CodeMirror-activeline-gutter";
  export default {
    name: "codeMirros",
    components: {
      codemirror
    },
    props: {
      value:{
        type: String,
        required: true,
        default: '',
      }
    },
    data() {
      return {
        codeMirrosOptions:{
		  // theme: 'solarized light',
          tabSize:4,
          readOnly:true, //只读
		  line:true,
          lineNumbers:true, //是否显示行数
		  mode: 'text/x-c++src',
		  // styleActiveLine: true,
        },
        resArr: '',
      };
    },
    methods: {
		lighterBg(start, end) {
			// console.log(this.$refs.codeMirros)
			// console.log(this.$refs.codeMirros.codemirror.getTextArea())
			let cm = this.$refs.codeMirros.codemirror;  //this.$refs.codeMirros.cminstance
			let active = [];
			this.clearActiveLines(cm);
			var start=Number(start);
			var end=Number(end);
			for (let i = start;i < end; i++){
				active.push(i);
			}
			cm.state.activeLines = active;
			for (var i = 0; i < active.length; i++) {
			  cm.addLineClass(active[i], "wrap", WRAP_CLASS);
			  cm.addLineClass(active[i], "background", BACK_CLASS);
			  cm.addLineClass(active[i], "gutter", GUTT_CLASS);
			}
			document.getElementsByClassName("CodeMirror-activeline")[0].scrollIntoView();
			
		},
		clearActiveLines(cm) {
		  if (!cm.state.activeLines) return 
		  for (var i = 0; i < cm.state.activeLines.length; i++) {
		    cm.removeLineClass(cm.state.activeLines[i], "wrap", WRAP_CLASS);
		    cm.removeLineClass(cm.state.activeLines[i], "background", BACK_CLASS);
		    cm.removeLineClass(cm.state.activeLines[i], "gutter", GUTT_CLASS);
		  }
		}
    }
  };
</script>

<style>
	.CodeMirror {
	            border: 1px solid #eee;
	            height: auto;
				/* line-height: 30px; */
	        }
	.CodeMirror-scroll {
	            height: auto;
	            overflow-y: hidden;
	            overflow-x: auto;
	        }
	.vue-codemirror{
		font-size: 25px;
	}
	        
</style>