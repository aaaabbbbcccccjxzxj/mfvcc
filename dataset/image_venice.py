from PIL import Image
import numpy as np
import h5py
from scipy.io import loadmat


class LoadData:
    @staticmethod
    def test_data(imgs_paths: list):
        imgs = []
        for path in imgs_paths:
            imgs.append(Image.open(path).convert('RGB'))
        targets = []
        for path in imgs_paths:
            temp = np.asarray(h5py.File(path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')['density'])
            targets.append(temp)
        rois = []
        for path in imgs_paths:
            temp = loadmat(path.replace('.jpg', '.mat').replace('images', 'ground_truth'))['roi']
            rois.append(temp)
        return imgs, targets, rois

    @staticmethod
    def train_data(imgs_paths: list, crop_size, scale=None):
        imgs, targets, rois = LoadData.test_data(imgs_paths)
        crop_size = (imgs[0].size[0] // 2 // 32 * 32, imgs[0].size[1] // 2 // 32 * 32) if not crop_size else (
            crop_size, crop_size)
        crop_max_x, crop_max_y = imgs[0].size[0] - crop_size[0] - 1, imgs[0].size[1] - crop_size[1] - 1
        # dx = int(scale['x'] * crop_size[0])
        # dy = int(scale['y'] * crop_size[1])
        dx = int(scale['x'] * crop_max_x)
        dy = int(scale['y'] * crop_max_y)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        for i in range(len(targets)):
            targets[i] = targets[i][dy: crop_size[1] + dy, dx: crop_size[0] + dx]
        for i in range(len(targets)):
            rois[i] = rois[i][dy: crop_size[1] + dy, dx: crop_size[0] + dx]
        if scale['flip'] >= .5:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            for i in range(len(targets)):
                targets[i] = np.fliplr(targets[i])
            for i in range(len(rois)):
                rois[i] = np.fliplr(rois[i])
        return imgs, targets, rois
