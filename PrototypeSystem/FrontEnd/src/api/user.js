import * as API from './index';
// API GET POST

export const register=(params)=>{
      return API.POST('/verify/register',params)
}
export const login=(params)=>{
      return API.POST('/verify/login',params)
}