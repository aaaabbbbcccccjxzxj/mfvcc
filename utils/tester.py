import functools
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from model.ddcm import DDCM
from utils.time_estimator import TimeEstimator
from utils.utils import Utils


class Tester(Utils):
    def __init__(self, args):
        super(Tester, self).__init__()
        print('Initializing tester ... ')
        self.args = args
        self.device = torch.device('cuda')
        self.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.time_estimator = TimeEstimator()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.manual_seed(self.args.seed)
        test_path = Path(self.args.dataset_path) / 'test' / 'images'
        self.test_list = [str(path) for path in test_path.glob('*.jpg')]
        cmp = self.cmp_v if self.video_dataset_mode == 'venice' else self.cmp
        self.test_list.sort(key=functools.cmp_to_key(cmp))
        self.test_list = [self.test_list]
        self.mask = self.get_roi_mask(self.args.mask_path)
        self.criterion = nn.MSELoss(reduction='sum').to(self.device)
        if os.path.isfile(self.args.pth_path):
            print(f'loading network args from {self.args.pth_path} ...')
            checkpoint = torch.load(self.args.pth_path)
            self.model = DDCM(is_train=False).to(self.device)
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise Exception(f'args not found at {self.args.pth_path}')
        if self.args.is_gray:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print('tester initializing done.')

    def execute(self):
        print('test begins ... ')
        return self.pre_test(self.test_list, False)
