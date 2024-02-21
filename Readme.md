pip install vitables vitables xxxx.h5
HDFVIEW

deblur first version:
1. blur input
2. eventloss like eventac or e2vid
better method: read both in one train loop, get a hrbrid loss to consider both blur and event
better method have finished, colmap dataset,with images(e2vid or other event to gray) and blurryImages

存在两个问题：1位姿能否和真实对上，要求imgaes两次训练读同一个 2 仿真event与真实event能否对上，判断方式如下：
如果e2vid选则前一个作为真值，那么eventloss的仿真做差要变成i+1 - i
如果e2vid选当前作为真值，那么eventloss的仿真做差要变成i - （i-1）

eventloss时间不应该过长，大部分时间不要启用虽然能训练很多轮，但似乎后边变化不大了
让C偏大，效果显著的好
最终训练时，崩溃的那几帧要排除掉，效果肯定更好！很简单，如果index =xx, index =1