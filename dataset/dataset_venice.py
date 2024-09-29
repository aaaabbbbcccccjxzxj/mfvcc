import math
import random
from itertools import chain
import cv2
import torch
from torch.utils.data import Dataset
from dataset.image_venice import LoadData


class BaseDataset(Dataset):
    def __init__(self, root, transform, crop_size=None, train=None):
        super().__init__()
        self.lines = list(chain(*root))
        self.rand_mask = list(range(len(self.lines)))
        self.crop_size = crop_size
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.rand_mask)

    def trans(self, imgs, targets, rois):

        target_new_shape = ((targets[0].shape[1] // 32) * 4, (targets[0].shape[0] // 32) * 4)
        # target_new_shape = ((targets[0].shape[1] // 8), (targets[0].shape[0] // 8))
        rate = (targets[0].shape[1] / target_new_shape[0]) * (targets[0].shape[0] / target_new_shape[1])
        for i in range(len(rois)):
            temp = cv2.resize(rois[i], target_new_shape)
            rois[i] = torch.tensor(temp).type(torch.FloatTensor)
        for i in range(len(targets)):
            temp = torch.tensor(cv2.resize(targets[i], target_new_shape) * rate).type(torch.FloatTensor)
            targets[i] = (temp * rois[i]).unsqueeze(0)
        for i in range(len(imgs)):
            imgs[i] = self.transform(imgs[i])
        # imgs:(3,h,w),targets:(1,h,w),mask:(h,w)
        return imgs, targets, rois

    @staticmethod
    def get_rand_scale():
        return {'x': random.random(), 'y': random.random(), 'flip': random.random()}


class Crowdataset_venice(BaseDataset):
    def __init__(self, root, transform, crop_size=None, train=None, f_num=3):
        super(Crowdataset_venice, self).__init__(root, transform, crop_size, train)
        self.f_num = f_num
        self.paths_index_len = len(self.rand_mask)
        self.rframe = self.paths_index_len % f_num
        self.rframe_t1 = 23 % f_num
        self.t1_len = math.ceil(23 / f_num)
        self.rframe_t2 = 27 % f_num
        self.t2_len = math.ceil(27 / f_num)
        self.rframe_t3 = 37 % f_num
        self.t3_len = math.ceil(37 / f_num)
        if self.train == 'train':
            for i in range(f_num - 1):
                self.rand_mask.pop()
            # if len(self.rand_mask) < 3000:
            #     self.rand_mask *= 2
            # elif len(self.rand_mask) < 100:
            #     self.rand_mask *= 25
            random.shuffle(self.rand_mask)
        elif self.train == 'val':
            temp = []
            for i in range(0, self.paths_index_len - self.rframe, f_num):
                temp.append(i)
            self.rand_mask = temp if self.rframe == 0 else temp + [self.paths_index_len - f_num]
        else:
            # 1 - 23, 24 - 50, 51 - 87
            temp_1 = []
            for i in range(0, 23 - self.rframe_t1, f_num):
                temp_1.append(i)
            front_1 = temp_1 if self.rframe_t1 == 0 else temp_1 + [23 - f_num]
            temp_2 = []
            for i in range(23, 50 - self.rframe_t2, f_num):
                temp_2.append(i)
            front_2 = temp_2 if self.rframe_t2 == 0 else temp_2 + [50 - f_num]
            temp_3 = []
            for i in range(50, 87 - self.rframe_t3, f_num):
                temp_3.append(i)
            front_3 = temp_3 if self.rframe_t3 == 0 else temp_3 + [87 - f_num]
            self.rand_mask = front_1 + front_2 + front_3
        # print(self.rand_mask)
        # for i in self.rand_mask:
        #     print(self.lines[i])

    def __getitem__(self, index):
        images_paths = []
        for i in range(self.f_num):
            images_paths.append(self.lines[self.rand_mask[index] + i])
        if self.train == 'train':
            scale = self.get_rand_scale()
            imgs, targets, rois = LoadData.train_data(images_paths, self.crop_size, scale)
        else:
            imgs, targets, rois = LoadData.test_data(images_paths)
        imgs, targets, rois = self.trans(imgs, targets, rois)
        # shape: imgs:(t,c,ch,cw) targets:(t,1,ch/8,cw/8) roi:(t,1,ch/8,cw/8)
        return torch.stack(imgs, dim=0), torch.stack(targets, dim=0), torch.stack(rois, dim=0).unsqueeze(1)
