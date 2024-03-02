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
可是火车又是i+1-i,真玄学.指标显示，i+1-i好像才是对的。。。。相信指标，眼睛真会骗人

eventloss时间不应该过长，大部分时间不要启用虽然能训练很多轮，但似乎后边变化不大了
让C偏大，效果显著的好
最终训练时，崩溃的那几帧要排除掉，效果肯定更好！很简单，如果index =xx, index =1
写论文时colmap模块要改成标定模块
目前的做法会让其棱角更分母，如有不妥，只进行最后一步训练即可
测试时同时把5,...等图一起输出了

真实数据应该也能做好，之前做不好还是因为位姿不对，rgb和event是不对应的，用rgb标的位姿给event用，问题比较大，用e2vid标还标不好，因此blender数据集很重要
两步式不仅能使其更容易找到最优点，而且速度也很快，不然即使weight给0，也很拖慢计算速度
训练可以统一记作8000+2000,事实上微调时2000后边也不咋变了
基本做完了，可以记录ssim,psnr,LPIPS了。至于怎样微调更好，就要根据实际表现了，训练一次不到7分钟，应该还好办
3000+1000 is better