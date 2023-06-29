
- 模型相较于transformer的三个出发点：
1. self-attention 机制的二次计算复杂度问题 o（L方）
2. 高内存使用量问题，对长序列输入堆叠时，总内存使用量高
3. 预测长输出时的速度骤降。动态解码速度如基于RNN一样慢

模型图：
![[Pasted image 20230627142936.png]]
多头稀疏化的注意力机制

- 核心创新点
  - ProbSparse self-attention: 概率稀疏注意力
     self-attention的权重构成长尾分布，即很少的权重贡献主要attention，其它可以被忽略。
     方法：
     1. 为每个query都随机采样部分的key， 默认值为5* InL
     2. 计算每个query的稀疏性得分M(qi, k). 选取稀疏性得分最高的N个query ，默认值为5* InL![[Pasted image 20230627145800.png]]
     3. 只计算N个query和key的点积结果，进而得到attention结果。 其余L-N个query则不计算，直接将self-attention 的输入取均值（mean（V））作为输出
	     - 整体时间复杂度为O(LlnL).
	     - query 稀疏性的准则：p(kj |qi )和均匀分布q的KL散度
	     - multi-head：每个head中得到的稀疏性最高的query也不相同
- 一步decoder
	 decoder的输入： Xde ∈ （Ltoken + Ly）* d. 
	 loss 只计算预测部分的
	 由两个DecoderLayer构成，包含：
	 -    mask self-attention: 将mask的点积变为 -∞，防止信息泄露
	 -    multi cross-attention：负责target sequence和source sequence的交互



## informer代码解读
