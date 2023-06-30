

- 20+21AAAI+Informer
    - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    - tags: #Attention; #LSTF; #Transformer; #ProbSparse; #Informer
    - [arxiv](https://arxiv.org/abs/2012.07436)
    - abs: 针对长时序预测问题 Long sequence time-series forecasting (LSTF). Transformer 的问题: quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. 因此 Informer 针对性地 1] ProbSparse self-attention mechanism 可以达到 O(L logL) 的 time complexity and memory usage. 2] 从而能够处理超长的输入; 3] generative style decoder 直接一次不需要序列生成, 加速. 