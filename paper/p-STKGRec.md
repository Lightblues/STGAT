
想法: 融入真实的POI KG

- 22KBS+STKG
    - Building and exploiting spatial–temporal knowledge graph for next POI recommendation
    - [acm](https://dl.acm.org/doi/10.1016/j.knosys.2022.109951); [github](https://github.com/WeiChen3690/STKGRec)
    - from [作者解读](https://mp.weixin.qq.com/s/qO9PHBwPRJPbe9keKA40LA); 更详细的 [note](https://mp.weixin.qq.com/s/NRcoAFQfwUaRjEmSsiV93g)
    - abs: 任务是POI预测, 难点在于数据稀疏性. 应用KG的难点: 1) 如何用KGs中的静态实体和关系表示用户的动态移动行为; 2) 如何利用KGs中不同类型的实体和关系来捕捉用户的长期和短期偏好. 本篇工作中, 直接从 check-in sequences 中构建 spatial–temporal KG (STKG), 其中 design a novel spatial–temporal transfer relation to intuitively capture users’ transition patterns between neighboring POIs. 在此基础上, 构建了 STKGRec, 建模 long- and short-term preferences of users (同时 spatial–temporal correlation of consecutive and nonconsecutive visits in the current check-in sequence). 

模型公式上比较复杂. 总的来看, 
- 构建STKG: (u,p) 之间通过访问时间点连边; (p,p) 之间通过用户的转移 $(\Delta t,\Delta d)$ 连边.
- 在STKG上进行TransR, 得到 u/p/t/st 的表示 (t表示用户访问时刻的边, st表示地点之间转移的边)
- 在此基础上, 构建不同的序列学习模型. 
    - 用户签到序列 $p'$
    - 引入用户的转移行为的序列 $z^{tr}$
- 长期: 汇总用户的签到序列, 得到长期偏好 $y^{+}$
- 短期: 
    - 时间上顺序 (consecutive)
    - 时空上顺序 (nonconsecutive)

## model

Spatial–temporal knowledge graph building

Spatial–temporal knowledge graph embedding
TransR

Long-term preference modeling with knowledge
- local spatial–temporal transfer representation of user u after the check-in POI $z^{tr} = p'+r^{st}$, (其中 $p' =e^u +r^t +e^p$ 表示了一个用户的签到行为) 表示「用户在地点p的转移行为」
- general spatial–temporal transfer representation $z^r = e^p+r^{st}$ 表示「地点p的通用转移行为」
- 经过 $g = Att(z^r, z^{tr},z^{tr})$ 得到用户转移序列的表示
- 对于g, 过一层GRU增强相关性, $h_{t} = GRU(g_t, h_{t-1})$, 得到「historical sequential dependency representation 」
- 类似的, 对于基本的签到行为 $p'$, 也过一层 $\tilde{h}_{t} = GRU(p'_t, \tilde{h}_{t-1})$
- 对于 $\tilde{h}$ 进行类似att的汇总, 得到用户的 **long-term preferences** $y^{+}$

Short-term preference modeling with knowledge
- 通过累加 $h^{st} = e^u + \sum e^{st}$ 得到用户经过多次转移行为后的状态. 
- 经过一层att得到用户 **consecutive spatial–temporal movemen** $y_n^{*}$
- 非连续情况下, 采用 Time-Geo-dilated GRU. 也即对seq $p'$ 中的每一个点, 从过去的长度限制$\gamma$的历史中筛选最相关的点, 经过 $h'_{t-1} = GRU(p'_{t-1}, h'_{t-\gamma})$ . 经过汇总得到 $y'_n$