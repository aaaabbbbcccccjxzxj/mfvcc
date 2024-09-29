import argparse
import csv
import multiprocessing
import os
import time
from utils.tester import Tester
from utils.trainer import Trainer

"""       ig       is_train       vd
ucsd     True   train val test  'ucsd'   resize to 952 × 632
canteen  False   True   False   'None'   
canteen  False   True   False   'None'
class    False   True   False   'None'
canteen  False   True   False   'None'
canteen  False  train val test  'canteen'   resize to 640 x 360
venice   False  train val test  'venice' 
"""

parser = argparse.ArgumentParser(description='DDCM train')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('save_path', metavar='SAVEPATH', help='the path to save checkpoint')
parser.add_argument('-sa', '--save_all', metavar='ISSAVE', help='is save all local', default=False)
parser.add_argument('-ld', '--load', metavar='LOAD', type=str, help='path to the checkpoint', default='None')
parser.add_argument('-gpu', metavar='GPU', type=str, help='gpu id to use', default='0')
parser.add_argument('-task', metavar='TASK', type=str, help='task id to use', default='None')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='fdst,venice,ucsd', default='None')
parser.add_argument('-ig', '--is_gray', metavar='IG', type=bool, help='is gray', default=False, )
# parser.add_argument('-pf', '--print_freq', metavar='PF', type=int, default=50, help='print frequency')
parser.add_argument('-sf', '--save_freq', metavar='SF', type=int, help='save frequency', default=None)
parser.add_argument('-ne', '--num_epoch', metavar='NE', type=int, help='num of epoch', default=None)
parser.add_argument('mask_path', metavar='MASKPATH', type=str, help='roi mask path')
parser.add_argument('-fn', '--frame_num', metavar='FN', type=int, help='frame num', default=None)
parser.add_argument('-lr', '--lr', metavar='LR', type=float, help='learning rate', default=1e-5)
parser.add_argument('-wd', '--weight_decay', metavar='WD', type=float, help='weight decay', default=1e-4)
parser.add_argument('-cs', '--crop_size', metavar='CS', type=int, default=None, help='crop size')
parser.add_argument('-cn', '--cluster_num', metavar='DM', type=int, help='divide train and val', default=1)
parser.add_argument('-bs', '--batch_size', metavar='BS', type=int, default=1, help='batch size')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    args.start_epoch = 0
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)
    trainer = Trainer(args)
    trainer.execute()

    # saved_path = args.save_path
    # wights = os.listdir(saved_path)
    #
    # mae_list = []
    # min_wight = ''
    # min_mae = 1000
    # min_rmse = 1000
    # for wight in wights:
    #     if os.path.splitext(wight)[-1] in ['.log', '.tar']:
    #         pass
    #     else:
    #         wight_path = os.path.join(saved_path, wight)
    #         args.pth_path = wight_path
    #         tester = Tester(args)
    #         mae, r_mse = tester.execute()
    #         if mae.item() < min_mae:
    #             min_mae = mae.item()
    #             min_wight = wight
    #             min_rmse = r_mse
    #         mae_list.append({wight: f'mae:{mae.item()}  rmse:{r_mse}'})
    # with open(f'{saved_path}/result.csv', 'a+', newline='') as f:
    #     writer = csv.writer(f)
    #     for res in mae_list:
    #         writer.writerow([res])
    #     writer.writerow(['--------------------------------------------------------------------------------------'])
    #     writer.writerow([f'best_result: {min_wight}:  mae:{min_mae}   rmse:{min_rmse}'])
