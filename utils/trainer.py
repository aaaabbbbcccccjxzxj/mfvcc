import functools
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from utils.logger import setlogger
from model.ddcm import DDCM
from utils.time_estimator import TimeEstimator
from utils.utils import Utils


class Trainer(Utils):
    def __init__(self, args):
        super(Trainer, self).__init__()
        print('Initializing trainer ... ')
        self.args = args
        self.device = torch.device('cuda')
        self.best_mae = 1e9
        self.local_mae = 1e9
        self.time_estimator = TimeEstimator()
        self.load = None if self.args.load == 'None' else self.args.load
        self.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.crop_size = self.args.crop_size
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        torch.cuda.manual_seed(self.args.seed)
        self.args.save_path = Path(self.args.save_path) / self.args.task
        self.args.save_path.mkdir(exist_ok=True)
        log_path = os.path.join(self.args.save_path, 'train.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        setlogger(log_path)
        if self.video_dataset_mode == 'fdst':
            train_path = Path(self.args.dataset_path) / 'train' / 'images'
            self.train_list = [str(path) for path in train_path.glob('*.jpg')]
            self.train_list.sort(key=functools.cmp_to_key(self.cmp))
            self.train_list = [self.train_list]
            self.args.print_freq = len(self.train_list[0]) - (self.args.frame_num - 1) * 60
            val_path = Path(self.args.dataset_path) / 'val' / 'images'
            self.val_list = [str(path) for path in val_path.glob('*.jpg')]
            self.val_list.sort(key=functools.cmp_to_key(self.cmp))
            self.val_list = [self.val_list]
        else:
            train_path = Path(self.args.dataset_path) / 'train' / 'images'
            train_list = [str(path) for path in train_path.glob('*.jpg')]
            train_list.sort(key=functools.cmp_to_key(self.cmp))
            self.train_list, self.val_list = Utils.divide_train_list(train_list, self.args.cluster_num)
            self.args.print_freq = len(self.train_list[0]) - self.args.frame_num + 1
        self.mask = self.get_roi_mask(self.args.mask_path)
        self.model = DDCM(is_train=True).to(self.device)
        self.criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        if self.load:
            if os.path.isfile(self.args.load):
                print(f'loading checkpoint from {self.load} ...')
                checkpoint = torch.load(self.load)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_mae = checkpoint['best_mae']
                self.local_mae = checkpoint['local_mae']
                print(f'successfully loaded checkpoint, epoch: {self.args.start_epoch}')
            else:
                print(f'checkpoint not found at {self.load}')
        self.save_freq = self.args.save_freq
        if self.args.is_gray:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for k, v in self.args.__dict__.items():
            logging.info("{}: {}".format(k, v))
        print('trainer initializing done.')

    def execute(self):
        suffixes = []
        for start in range(0, self.args.num_epoch, self.save_freq):
            end = min(start + self.save_freq - 1, self.args.num_epoch - 1)
            suffixes.append(f"{start}-{end}")
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            logging.info(f'epoch: {epoch} train begin ... ')
            # if epoch > 0 and epoch % 50 == 0:
            #    self.args.lr *= 0.9
            #    self.optimizer.param_groups[-1]['lr'] = self.args.lr
            logging.info("lr: {:.0e}".format(self.optimizer.param_groups[-1]['lr']))
            self.time_estimator.mark()
            self.time_estimator.estimate(epoch, self.args.num_epoch)
            self.pre_train(self.train_list, epoch)
            # save checkpoints
            Utils.save_ckpt_or_bestmodel(
                {
                    'epoch': epoch + 1,
                    'best_mae': self.best_mae,
                    'local_mae': self.local_mae,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },
                True,
                self.args.save_path,
                epoch,
                None,
                None
            )
            logging.info('validation begins ...')
            precision, _ = self.pre_test(self.val_list)
            is_best = precision < self.best_mae
            is_local_best = precision < self.local_mae
            self.best_mae = min(precision, self.best_mae)
            self.local_mae = min(precision, self.local_mae)
            logging.info(f' * local best MAE: {self.local_mae:.5f}')
            logging.info(f' * global best MAE: {self.best_mae:.5f}')
            if is_local_best:
                logging.info(f'Save model...epoch = {epoch:03d}')
                if self.args.save_all:
                    Utils.save_ckpt_or_bestmodel({'model': self.model.state_dict()}, False, self.args.save_path, epoch,
                                                 f'{epoch:03d}', is_best)
                else:
                    Utils.save_ckpt_or_bestmodel({'model': self.model.state_dict()}, False, self.args.save_path, epoch,
                                                 suffixes[epoch // self.save_freq], is_best)
            if epoch != 0 and epoch % self.save_freq == 0:
                self.local_mae = 1e9
