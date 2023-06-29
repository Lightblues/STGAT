
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
