



## 模型设计层面
1. informer中关于id的embedding是怎么建模的  
   - 在STGAT中，用户i在t时刻的位置encoder embedding：mit 被认为是用户embedding；通过与其它用户的embedding做attention获得mit的更新，再输出到decoder中

2. 空间attn怎么融入informer的embedding中
	- 更新encoder当前步的结果即可
   
3. id attn的设计方法
   - 参考GAT 的方法 


## 模型实现层面
STGAT 模型记录：
