
# MTL 多任务学习

code: https://github.com/QunBB/DeepLearning 其中多任务部分

## 基准: Shared-Bottom 方案

最naive的方案: 底层sheared MLP, 上面各子任务的MLP层. 
问题: 1] 原生MTL模型往往无法比task单独建模的效果好; 2] 对task之间的数据分布差异和相关性很敏感; 3] task之间固有的冲突会影响原声MTL模型的表现
原因: 1] 底层共享的参数容易偏向于某个task，或者说偏向于某个task的全局(局部)最优方向; 2] 不同task的梯度冲突，存在参数撕扯，比如两个task的梯度是正负相反，那最终的梯度可能被抵消为0，导致共享参数更新缓慢


## 18KDD+MMoE

- 18KDD+MMoE+ #Google
    - Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    - abs: 只出了MTL的效果受制于任务之间的relationships. 提出 Multi-gate Mixture-of-Experts (MMoE), 来显式学习任务之间的关系! 通过MoE来共享学习多任务, 对于每个任务用一个 gating network! 实验中, 1] 在合成的可以控制相关性的数据上进行了实验 (control the task relatedness), 证明了在若相关任务上表现更好; 2] 真实数据. 

总结
1.  对比Shared-Bottom模型，MMoE将底层的共享网络层拆分为多个共享的Expert，并且通过引入Gate来学习每个Expert对不同task的贡献程度；
2.  对应不同相关性的task，MMoE模型的效果比较稳定。这主要是因为相关性弱的task，可以通过Gate来利用不同的Expert。


## 20RecSys+PLE

引入: MMoE在弱相关性task中表现地相对比较稳定，但由于底层的Expert仍然是共享的（虽然引入Gate来让task选择Expert），所以还是会存在"跷跷板"的情况：一个task的效果提升，会伴随着另一个task的效果降低。
腾讯在2020的论文中，就对MMoE进行改进，提出了CGC（Customized Gate Control）、PLE（Progressive Layered Extraction）. 其实，从结构上来看，CGC可以认为是单层的PLE。

- 20RecSys+PLE+ #Tencent, #best-paper
    - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
    - [semantic](https://www.semanticscholar.org/paper/Progressive-Layered-Extraction-(PLE)%3A-A-Novel-(MTL)-Tang-Liu/0182197c3996d11867ef650b66a2ddf1efa6f631) 100+
    - note [腾讯的 (PLE) 为什么能获得RecSys2020最佳长论文奖？](https://www.zhihu.com/question/425243050); [多任务学习MTL模型：MMoE、PLE](https://zhuanlan.zhihu.com/p/425209494)
    - abs: MTL多任务模型的问题 1] negative transfer due to the complex and competing task correlation in real-world RS; 2] 跷跷板现象 (seesaw), 一个任务损害了另一个. 提出了 Progressive Layered Extraction (PLE), 模型要点: separates shared components and task-specific components explicitly and adopts a progressive routing mechanism to extract and separate deeper semantic knowledge gradually, improving efficiency of joint representation learning and information routing across tasks in a general setup. 核心在腾讯视频上进行了应用, 超过SOTA; 并且在公开任务上验证了. 

### CGC

跟MMoE的差别就在于：**除了共享的Expert之外，还加入了每个task自己的Specific Expert**

1.  所有task共享的expert（如上图Experts Shared）、每个task自己的expert（如上图task A的Experts A），跟MMoE一样，Expert也是模型输入Input映射而来：ReLU激活函数的全连接层；
2.  每个task通过Input映射为自己的Gate：一个没有bias和激活函数的全连接层，然后接softmax，即图中的Gating Network；
3.  每个task选择共享的Experts和task自己的Experts，通过task自己的Gate来得到多个Expert的加权平均，然后输入到task对应的Tower层（MLP网络层）；
4.  最后，通过对应task的Tower层输出，计算得到task的预测值。


### PLE

PLE其实可以认为是多层的CGC：

1.  由多个Extraction Network组成，每个Extraction Network就是CGC网络层，做法与CGC一致；
2.  第一层Extraction Network的输入是原生模型输入Input；
3.  但后面的Extraction Network，输入就不再是Input，而是所有Gate与Experts的加权平均的融合，这里的融合一般做法包括：拼接、加法融合、乘法融合，或者这三种的组合；
4.  最后一层Extraction Network中gate的数量等于task的数量，对应每个task的gate；
5.  而前面层的Extraction Network中gate的数量是task的数量+1，这里其实就是对应每个task的gate，加上共享expert的gate。


## 21CIKM+STAR

- 21CIKM+STAR+ #Ali
    - One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
    - [arxiv](https://arxiv.org/abs/2101.11427); 
    - note [论文精读](https://zhuanlan.zhihu.com/p/543243471)



- SAR-Net #Ali
    - SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios
    - [arxiv](https://arxiv.org/abs/2110.06475); 
    - [阿里巴巴SAR-Net阅读记载](https://zhuanlan.zhihu.com/p/539958163)
