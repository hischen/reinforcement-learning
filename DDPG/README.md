## DDPG
    
* 在连续控制领域，比较经典的强化学习算法就是深度确定性策略梯度(deep deterministic policy gradient，DDPG)，DDPG 的特点可以从它的名字中拆解出来，拆解成深度、确定性和策略梯度。  
* 深度是因为用了神经网络;确定性表示 DDPG 输出的是一个确定性的动作，可以用于有连续动作的 环境;策略梯度代表的是它用到的是策略网络。REINFORCE 算法每隔一个回合就更新一次，但 DDPG 是每个步骤都会更新一次策略网络，它是一个单步更新的策略网络。  
* DDPG 是深度 Q 网络的一个扩展版本，可以扩展到连续动作空间。在 DDPG 的训练中，它借鉴了 深度 Q 网络的技巧:目标网络和经验回放。经验回放与深度 Q 网络是一样的，
但目标网络的更新与深度 Q 网络的有点儿不一样。提出 DDPG 是为了让深度 Q 网络可以扩展到连续的动作空间，就是我们刚才提 到的小车速度、角度和电压等这样的连续值。
DDPG 在深度 Q 网络基础上加了一个策略 网络来直接输出动作值，所以 DDPG 需要一边学习 Q 网络，一边学习策略网络。Q 网络的参数用 w 来 表示。策略网络的参数用 θ 来表示。
我们称这样的结构为演员-评论员的结构。
* 深度确定性策略梯度(DDPG):“四不像”的方法:训练时有策略网络和价值网络，预估时只有策略网络 
* 存在DQN 的高估问题：最大化造成高估，自举造成偏差传播
第一，TD 目标是对真实动作价值的高估;第二，自举导致高估的传播



## TD3
* 高估问题的解决方案
  * 目标网络:两个价值网络、一个策略网络各构建一个目标网络，它们与价值网络、策略网络的结构完全相同，但是参数不同
  * 截断双 Q 学习(clipped double Q-learning)：使用两个价值网络和一个策略网络，参考[https://datawhalechina.github.io/easy-rl/#/chapter12/chapter12]   和DRL page158
* 往动作中加噪声:引入了平滑化（smoothing）思想。TD3在目标动作中加入噪声，通过平滑 Q 沿动作的变化，使策略更难利用 Q 函数的误差。
* 延迟的策略更新（delayed policy updates):减小更新策略网络和目标网络的频率,较低的频率更新动作网络，以较高的频率更新评价网络，通常每更新两次评价网络就更新一次策略

    