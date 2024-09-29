import random
from itertools import chain
import cv2
import torch
from torch.utils.data import Dataset
from dataset.image import LoadData


class BaseDataset(Dataset):
    def __init__(self, root, transform, mask, crop_size=None, train=None):
        super().__init__()
        self.lines = list(chain(*root))
        self.rand_mask = list(range(len(self.lines)))
        self.crop_size = crop_size
        self.transform = transform
        self.mask = mask
        self.train = train

    def __len__(self):
        return len(self.rand_mask)

    def trans(self, imgs, targets, mask):

        target_new_shape = ((targets[0].shape[1] // 32) * 4, (targets[0].shape[0] // 32) * 4)
        rate = (targets[0].shape[1] / target_new_shape[0]) * (targets[0].shape[0] / target_new_shape[1])
        if not isinstance(mask, type(None)):
            mask = cv2.resize(mask, target_new_shape)
            mask = torch.tensor(mask)
        else:
            mask = torch.ones(target_new_shape[1], target_new_shape[0])
        for i in range(len(targets)):
            temp = torch.tensor(cv2.resize(targets[i], target_new_shape) * rate).type(torch.FloatTensor)
            targets[i] = (temp * mask).unsqueeze(0)
        for i in range(len(imgs)):
            imgs[i] = self.transform(imgs[i])
        # imgs:(3,h,w),targets:(1,h,w),mask:(h,w)
        return imgs, targets, mask

    @staticmethod
    def get_rand_scale():
        return {'x': random.random(), 'y': random.random(), 'flip': random.random()}


class Crowdataset_ucsd(BaseDataset):
    def __init__(self, root, transform, mask, crop_size=None, train=None, f_num=3):
        super(Crowdataset_ucsd, self).__init__(root, transform, mask, crop_size, train)

        self.f_num = f_num
        self.paths_index_len = len(self.rand_mask)
        self.rframe = self.paths_index_len % f_num
        self.rframe_half = self.paths_index_len // 2 % f_num
        if self.train == 'train':
            # if len(self.rand_mask) < 3000:
            #     self.rand_mask *= 2
            # elif len(self.rand_mask) < 100:
            #     self.rand_mask *= 25
            for i in range(f_num - 1):
                self.rand_mask.pop()
            random.shuffle(self.rand_mask)
        elif self.train == 'val':
            temp = []
            for i in range(0, self.paths_index_len - self.rframe, f_num):
                temp.append(i)
            self.rand_mask = temp if self.rframe == 0 else temp + [self.paths_index_len - f_num]

        else:
            # 1 - 600, 1401 - 2000
            temp_f = []
            for i in range(0, self.paths_index_len // 2 - self.rframe_half, f_num):
                temp_f.append(i)
            front_r = temp_f if self.rframe_half == 0 else temp_f + [self.paths_index_len // 2 - f_num]
            temp_b = []
            for i in range(self.paths_index_len // 2, self.paths_index_len - self.rframe_half, f_num):
                temp_b.append(i)
            front_b = temp_b if self.rframe_half == 0 else temp_b + [self.paths_index_len - f_num]
            self.rand_mask = front_r + front_b
        # print(self.rand_mask)
        # for i in self.rand_mask:
        #     print(self.lines[i])

    def __getitem__(self, index):
        images_paths = []
        for i in range(self.f_num):
            images_paths.append(self.lines[self.rand_mask[index] + i])
        mask = self.mask
        if self.train == 'train':
            scale = self.get_rand_scale()
            imgs, targets, mask = LoadData.train_data(images_paths, self.crop_size, scale, mask)
        else:
            imgs, targets = LoadData.test_data(images_paths)
        imgs, targets, mask = self.trans(imgs, targets, mask)
        # shape: imgs:(t,c,ch,cw) targets:(t,1,ch/8,cw/8) mask:(1,ch/8,cw/8)
        return torch.stack(imgs, dim=0), torch.stack(targets, dim=0), mask.unsqueeze(0)
