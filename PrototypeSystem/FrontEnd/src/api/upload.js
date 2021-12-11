import * as API from './index';

export const postFile=(params)=>{
	return API.POST('/upload/upload_package', params)
}

export const getRecords=()=>{
	return API.POST('/upload/get_project_list')
}