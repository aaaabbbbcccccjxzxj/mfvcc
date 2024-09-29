import logging
import math
import os
import re
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset_bccm import Crowdataset
from dataset.dataset_fdst import Crowdataset_fdst
from dataset.dataset_ucsd import Crowdataset_ucsd
from dataset.dataset_venice import Crowdataset_venice


class Utils:
    def __init__(self):
        super().__init__()
        self.args = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.transform = None
        self.mask = None
        self.criterion = None
        self.video_dataset_mode = None
        self.crop_size = None
        self.time_estimator = None

    @staticmethod
    def save_ckpt_or_bestmodel(state, is_ckpt: bool, save_path, epoch, suffix, is_best):
        if is_ckpt:
            ready_save_ckpt = os.path.join(save_path, f'ckpt_{epoch}.tar')
            torch.save(state, ready_save_ckpt)
            previous_ckpt = os.path.join(save_path, f'ckpt_{epoch - 1}.tar')
            if os.path.exists(previous_ckpt):
                os.remove(previous_ckpt)
        else:
            local_model_wight = os.path.join(save_path, f'local_m_{suffix}.pth')
            torch.save(state, local_model_wight)
            if is_best:
                best_model_wight = os.path.join(save_path, 'best_model.pth')
                shutil.copyfile(local_model_wight, best_model_wight)

    @staticmethod
    def cmp(x, y):
        a = int(re.findall(r'(\d+)\.\w+$', str(x))[0])
        b = int(re.findall(r'(\d+)\.\w+$', str(y))[0])
        return -1 if a < b else 1

    @staticmethod
    def cmp_v(x, y):
        x_num, x_suffix = re.findall(r'(\d+)_(\d+)\.\w+$', str(x))[0]
        y_num, y_suffix = re.findall(r'(\d+)_(\d+)\.\w+$', str(y))[0]
        if int(x_num) == int(y_num):
            return int(x_suffix) - int(y_suffix)
        else:
            return int(x_num) - int(y_num)

    @staticmethod
    def divide_train_list(train_list, num):
        val_list = []
        new_train_list = []
        val_ele_len = len(train_list) // (10 * num)
        divide_points = [(len(train_list) * i) // num for i in range(num + 1)]
        for i in range(num):
            val_list.append(train_list[divide_points[i + 1] - val_ele_len: divide_points[i + 1]])
            new_train_list.append(train_list[divide_points[i]: divide_points[i + 1] - val_ele_len])
        return new_train_list, val_list

    @staticmethod
    def get_roi_mask(mask_path):
        if mask_path != 'None':
            mask = np.load(mask_path)
            mask[mask <= 1e-4] = 0
        else:
            mask = None
        return mask

    def unpack_data(self, data):
        images, targets, mask = data
        images = images.to(self.device)
        targets = targets.squeeze(0).to(self.device)
        mask = mask.to(self.device) if self.video_dataset_mode != 'venice' else mask.squeeze(0).to(self.device)
        return images, targets, mask

    def computer_(self, i, data_loader_len, res, targets, mae, r_mse, rframe):
        if i + 1 != data_loader_len:
            for f in range(self.args.frame_num):
                mae += abs(res[f].data.sum() - targets[f].sum().type(torch.FloatTensor).to(self.device))
                r_mse += (res[f].data.sum() - targets[f].sum().type(torch.FloatTensor).to(self.device)) ** 2
        else:
            if rframe == 0:
                for f in range(self.args.frame_num):
                    mae += abs(res[f].data.sum() - targets[f].sum().type(torch.FloatTensor).to(self.device))
                    r_mse += (res[f].data.sum() - targets[f].sum().type(torch.FloatTensor).to(self.device)) ** 2
            else:
                hascal = self.args.frame_num - rframe
                for f in range(rframe):
                    mae += abs(
                        res[f + hascal].data.sum() - targets[f + hascal].sum().type(torch.FloatTensor).to(self.device))
                    r_mse += (res[f + hascal].data.sum() - targets[f + hascal].sum().type(torch.FloatTensor).to(
                        self.device)) ** 2
        return mae, r_mse

    # train
    def pre_train(self, data_list, epoch):
        if self.video_dataset_mode == 'ucsd':
            data_loader = DataLoader(
                Crowdataset_ucsd(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train='train',
                    f_num=self.args.frame_num
                ),
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        elif self.video_dataset_mode == 'fdst':
            data_loader = DataLoader(
                Crowdataset_fdst(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train='train',
                    f_num=self.args.frame_num
                ),
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        elif self.video_dataset_mode == 'venice':
            data_loader = DataLoader(
                Crowdataset_venice(
                    data_list,
                    self.transform,
                    crop_size=self.crop_size,
                    train='train',
                    f_num=self.args.frame_num
                ),
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            data_loader = DataLoader(
                Crowdataset(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train=True,
                    f_num=self.args.frame_num
                ),
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        self.model.train()
        loss_sum = 0
        mae = 0
        self.time_estimator.simple_mark()
        for i, data in enumerate(data_loader):
            images, targets, mask = self.unpack_data(data)
            # images: (1, f, c, h, w) targets: (f, 1, h, w)  remove batch dim in unpack function
            res = self.model(images)
            # res: (f, 1, h, w) mask:(1,1,h,w) or (f, 1, h, w)
            # res do mask, targets have did it.
            res *= mask
            loss = self.criterion(res, targets)
            loss_sum += loss
            mae += abs(res.data.sum() - targets.sum().type(torch.FloatTensor).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print each epoch
            if (i + 1) % self.args.print_freq == 0:
                loss_sum /= self.args.print_freq * self.args.batch_size * self.args.frame_num
                mae /= self.args.print_freq * self.args.batch_size * self.args.frame_num
                logging.info(f'epoch {epoch:<3}: [{i + 1:>5}/{len(data_loader)}]batch loss: {loss_sum:<16.13f}'
                             f' mae: {mae:<8.5f} time: {self.time_estimator.query_time_span()}s')
                loss_sum = 0
                mae = 0
                self.time_estimator.simple_mark()

    # test
    def pre_test(self, data_list, is_val=True):
        if self.video_dataset_mode == 'ucsd':
            data_loader = DataLoader(
                Crowdataset_ucsd(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train='val' if is_val else 'test',
                    f_num=self.args.frame_num
                ),
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        elif self.video_dataset_mode == 'fdst':
            data_loader = DataLoader(
                Crowdataset_fdst(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train='val' if is_val else 'test',
                    f_num=self.args.frame_num
                ),
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        elif self.video_dataset_mode == 'venice':
            data_loader = DataLoader(
                Crowdataset_venice(
                    data_list,
                    self.transform,
                    crop_size=self.crop_size,
                    train='val' if is_val else 'test',
                    f_num=self.args.frame_num
                ),
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            data_loader = DataLoader(
                Crowdataset(
                    data_list,
                    self.transform,
                    self.mask,
                    crop_size=self.crop_size,
                    train=False,
                    f_num=self.args.frame_num
                ),
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        self.model.eval()
        mse = 0
        mae = 0
        r_mse = 0
        if self.video_dataset_mode is None or (
                self.video_dataset_mode == 'ucsd' and data_loader.dataset.train == 'val') or (
                self.video_dataset_mode == 'venice' and data_loader.dataset.train == 'val'):
            rframe = data_loader.dataset.rframe
        elif self.video_dataset_mode == 'ucsd' and data_loader.dataset.train == 'test':
            rframe_half = data_loader.dataset.rframe_half
        elif self.video_dataset_mode == 'fdst':
            rframe = data_loader.dataset.rframe
        else:
            rframe_t1 = data_loader.dataset.rframe_t1
            one_c_t1 = data_loader.dataset.t1_len
            rframe_t2 = data_loader.dataset.rframe_t2
            one_c_t2 = data_loader.dataset.t2_len + one_c_t1
            rframe_t3 = data_loader.dataset.rframe_t3
            one_c_t3 = data_loader.dataset.t3_len + one_c_t2
        self.time_estimator.simple_mark()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print(f' {((i + 1) / len(data_loader)) * 100:.1f}% ...\r', end='')
                images, targets, mask = self.unpack_data(data)
                res = self.model(images)
                res *= mask
                mse += self.criterion(res, targets).item()
                if self.video_dataset_mode is None or (
                        self.video_dataset_mode == 'ucsd' and data_loader.dataset.train == 'val') or (
                        self.video_dataset_mode == 'venice' and data_loader.dataset.train == 'val'):
                    mae, r_mse = self.computer_(i, len(data_loader), res, targets, mae, r_mse, rframe)
                elif self.video_dataset_mode == 'ucsd' and data_loader.dataset.train == 'test':
                    one_c = len(data_loader) // 2
                    mae, r_mse = self.computer_(i, one_c * math.ceil((i + 1) / one_c), res, targets, mae, r_mse,
                                                rframe_half)
                elif self.video_dataset_mode == 'fdst':
                    one_c = len(data_loader) // 60 if data_loader.dataset.train == 'val' else 40
                    mae, r_mse = self.computer_(i, one_c * math.ceil((i + 1) / one_c), res, targets, mae, r_mse, rframe)
                else:
                    if i + 1 <= one_c_t1:
                        mae, r_mse = self.computer_(i, one_c_t1, res, targets, mae, r_mse, rframe_t1)
                    elif one_c_t1 < i + 1 <= one_c_t2:
                        mae, r_mse = self.computer_(i, one_c_t2, res, targets, mae, r_mse, rframe_t2)
                    else:
                        mae, r_mse = self.computer_(i, one_c_t3, res, targets, mae, r_mse, rframe_t3)
        if self.video_dataset_mode == 'ucsd' and data_loader.dataset.train == 'test':
            mae /= 1200
            mse /= 1200
            r_mse /= 1200
        elif self.video_dataset_mode == 'fdst':
            mae /= 900 if data_loader.dataset.train == 'val' else 6000
            mse /= 900 if data_loader.dataset.train == 'val' else 6000
            r_mse /= 900 if data_loader.dataset.train == 'val' else 6000
        elif self.video_dataset_mode == 'venice' and data_loader.dataset.train == 'test':
            mae /= 87
            mse /= 87
            r_mse /= 87
        else:
            mae /= (len(data_loader) - 1) * self.args.frame_num + (self.args.frame_num if rframe == 0 else rframe)
            mse /= (len(data_loader) - 1) * self.args.frame_num + (self.args.frame_num if rframe == 0 else rframe)
            r_mse /= (len(data_loader) - 1) * self.args.frame_num + (self.args.frame_num if rframe == 0 else rframe)
        r_mse = float(r_mse) ** .5
        if is_val:
            logging.info(f' MAE: {mae:.5f}')
            logging.info(f' MSE: {mse:.5f}')
            logging.info(f' RMSE: {r_mse:.5f}')
            logging.info(f' Time: {self.time_estimator.query_time_span()}s')
        else:
            print(f' MAE: {mae:.5f}')
            print(f' MSE: {mse:.5f}')
            print(f' RMSE: {r_mse:.5f}')
            print(f' Time: {self.time_estimator.query_time_span()}s')
        return mae, r_mse
