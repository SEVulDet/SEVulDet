import * as API from './index';

export const getlist=(params)=>{
	return API.POST('/showCode/getProjectTree', params)
}

export const getfile=(params)=>{
	return API.POST('/showCode/getFileRes', params)
}

export const getfilecode=(url,param)=>{
	let params = {
		param: param,
		headers: {'Content-Type': 'application/x-www-form-urlencoded,charset=utf-8'}
	}
	return API.GET(url, params)
}

// export const getfilecode=(url,param)=>{
// 	let params = {
// 		param: param,
// 	}
// 	return API.GET(url, params)
// }
