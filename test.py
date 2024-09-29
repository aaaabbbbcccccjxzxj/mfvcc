import argparse
import time
import multiprocessing

from utils.tester import Tester

parser = argparse.ArgumentParser(description='Single test')
parser.add_argument('-dataset_path', metavar='DATASET', help='path to dataset', default="../bccmfvu/bus")
parser.add_argument('-pth_path', metavar='PTH', help='path to pth file', default="./wight/bus_t6.pth")
parser.add_argument('-gpu', metavar='GPU', type=str, help='gpu id to use', default="0")
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path',
                    default='../bccmfvu/bus/bus_roi.npy')
parser.add_argument('-ig', '--is_gray', metavar='IG', type=bool, default=False, help='is gray')
parser.add_argument('-fn', '--frame_num', metavar='FN', type=int, default=6, help='frame num')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)
    tester = Tester(args)
    tester.execute()

