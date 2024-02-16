pip install vitables vitables xxxx.h5
HDFVIEW

deblur first version:
1. blur input
2. eventloss like eventac or e2vid
better method: read both in one train loop, get a hrbrid loss to consider both blur and event
better method have finished, colmap dataset,with images(e2vid or other event to gray) and blurryImages