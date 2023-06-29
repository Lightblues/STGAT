
STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction
[ieee](https://ieeexplore.ieee.org/document/9010834)
[论文翻译](https://blog.csdn.net/Sun_ZD/article/details/110916188); [源码部分解析 (code analysis)](https://blog.csdn.net/u010730851/article/details/106580342)

Input: obs_traj_rel
obs_traj_rel+pred_traj_gt_rel

## TrajectoryGenerator

obs_traj_rel: [obsL, sum_len, xy] = [8, 1413, 2]
可以理解成LSTM的输入, 但每个sample的维度是 [8, 2]

拆成了8个 input_t: [1413, 2], 可以作为 traj_lstm_model 的输入了!

    input_t0    input_t1
(h0,c0) -> (h1,c1) -> (h2,c2) -> ...

### stage1
input[obs_traj_rel] -> output[pred_traj_rel]
形状不变!
目标: 用obs_traj_rel预测pred_traj_rel, 要求保持不变!
为什么? 训练 M-LSTM 里面的参数

X = obs_traj_rel
y_hat = M-LSTM[X; paras]
y = X
loss = l2(y, y_hat)

<!-- y = X
y_hat = Encoder(Decoder(X))
loss = l2(y, y_hat) -->

### stage2

gatencoder = GATEncoder()

训练 M-LSTM, GAT, G-LSTM

### stage3


我爱你: I love you

no teacher-force
你 -> I
I -> like
like -> ..

no teacher-force
你 -> I
I -> like
love -> ..


1,2,...10
1,2.....20

start_end
[0,9],[10,39]

1.空间 attn是怎么把邻居信息传入的
2.transformer相比于Istm需要修改什么，简化什么
3.weather回测任务交付
4.整理出行物品:一次性浴巾、隐形眼镜等。16:00必须打车出发
5.找mentor聊:1) 后续工作以模型为重，希望有挑战性的任务能更体现出个人能力。2) 在实际工作中模型相关的工作到底有多重要?算法工程师的核心竞争力在哪里3) 答辩的时候怎么突出个人的贡献(过往答辩的案例)感觉以模型为主的工作可以强调设计思路实现方法以及成果;以业务为主的工作仙乎很难讲出亮点(感觉像对于新任务套现成的方案或者加加特征这种会有点水)


