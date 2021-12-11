import axios from 'axios';
import NProgress from 'nprogress';//请求进度条


axios.defaults.baseURL = '/api';
// axios.defaults.headers.post['Content-Type'] = 'audio/x-wav';
// axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';

// 添加请求拦截器
axios.interceptors.request.use(function (config) {
    // 在发送请求之前做些什么
    NProgress.start()//开始加载进度条
    return config;
  }, function (error) {
    // 对请求错误做些什么
    return Promise.reject(error);
  });

// 添加响应拦截器
axios.interceptors.response.use(
	response => {
		// 对响应数据做点什么
		
		NProgress.done()  // 进度条完成

		if (response.status == 200 || response.data.status === 200) {

			return Promise.resolve(response);        
		} else {
		    return Promise.reject(response);        
		}    
	},
	error => {
        if (error.response.status) {            
            switch (error.response.status) {                
                // 401: 未登录
                // 未登录则跳转登录页面，并携带当前页面的路径
                // 在登录成功后返回当前页面，这一步需要在登录页操作。                
                case 401:                    
                    router.replace({                        
                        path: '/login',                        
                        query: { 
                            redirect: router.currentRoute.fullPath 
                        }
                    });
                    break;

                // 403 token过期
                // 登录过期对用户进行提示
                // 清除本地token和清空vuex中token对象
                // 跳转登录页面                
                case 403:
                     Toast({
                        message: '登录过期，请重新登录',
                        duration: 1000,
                        forbidClick: true
                    });
                    // 清除token
                    localStorage.removeItem('token');
                    store.commit('loginSuccess', null);
                    // 跳转登录页面，并将要浏览的页面fullPath传过去，登录成功后跳转需要访问的页面 
                    setTimeout(() => {                        
                        router.replace({                            
                            path: '/login',                            
                            query: { 
                                redirect: router.currentRoute.fullPath 
                            }                        
                        });                    
                    }, 1000);                    
                    break; 

                // 404请求不存在
                case 404:
                    Toast({
                        message: '网络请求不存在',
                        duration: 1500,
                        forbidClick: true
                    });
                    break;
                // 其他错误，直接抛出错误提示
                default:
                    // Toast({
                    //     message: error.response.data.message,
                    //     duration: 1500,
                    //     forbidClick: true
                    // });
            }
            return Promise.reject(error.response);
        }
    }
  //  function (error) {
  //   // 对响应错误做点什么
  //   return Promise.reject(error);
  // }
  );



export const GET=(url,params)=>{
  // return axios.get(`${baseUrl}${url}`,{params:params}).then(data=>data)
  if (params.headers){
	  return axios.get(url,{
	  						params:params.param,
	  						headers:params.headers,
	  						}).then(data=>data)
  }else{
		return axios.get(url,{
	  						params:params.param
	  						}).then(data=>data)
  }
}

export const POST=(url,params)=>{
  // return axios.post(`${baseUrl}${url}`,params).then(data=>data);
  return axios.post(url,params).then(data=>data);
}
