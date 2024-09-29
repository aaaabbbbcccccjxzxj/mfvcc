# DACM
official code for Dual-branch Adjacent Connection and Channel Mixing Network for Video Crowd Counting. 

## Dataset
- Bus-images: [BaiduNetDisk](https://pan.baidu.com/s/15zncl2jPMnX_1RgiS_ejww?pwd=vp7j).
- Bus-ground truth: [BaiduNetDisk](https://pan.baidu.com/s/1-NNZvIxcLeVjnEmaQfAi7Q?pwd=5slv).

- Canteen-images: [BaiduNetDisk](https://pan.baidu.com/s/1ze84yTUw-Y2-Z5sUUzWnng?pwd=j17o). 
- Canteen-ground truth: [BaiduNetDisk](https://pan.baidu.com/s/10b_H4sZCN_NORDVRJ0yQvg?pwd=b41u). 

## Pretraining Weight
- ShiftVit-Tiny: [ShiftVit-T-G.pth](https://pan.baidu.com/s/1faf5lFmemvptlAaYd19EZw?pwd=ftvt).

## Install dependencies
torch >= 1.0, torchvision, opencv, numpy, scipy, etc.

##  Take training and testing of Bus dataset for example:
1. Download Bus-images.
2. Download Bus-ground truth.
3. Set the folder structure should look like this:
```shell 
Bus
├──train
    ├──ground_truth    
        ├──xxx_0.h5
        ├──xxx_10.h5
        ├──....
    ├──images
        ├──xxx_0.jpg
        ├──xxx_10.jpg
├──test
    ├──ground_truth
    ├──images
├──bus_roi.npy
```
4. Download Pretraining weight ShiftVit-T-G.pth.
5. Create a folder named 'wight' in the project directory and put the pth file in it.
6. Train Bus.
```shell 
python train.py (dataset path) (directory for saving model weights) (roi path)
```
7. Test Bus.
```shell 
python auto_test.py (dataset path) (directory for saving model weights) (roi path)
```
