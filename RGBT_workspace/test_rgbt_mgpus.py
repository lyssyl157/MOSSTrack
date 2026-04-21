import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import importlib

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

prj = join(dirname(__file__), '../..')
if prj not in sys.path:
    sys.path.append(prj)

import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if
                             os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted(
            [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted(
            [seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(RGB_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(RGB_gt[:, [1, 3]], axis=1)[:, None]
        RGB_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        x_min = np.min(T_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(T_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(T_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(T_gt[:, [1, 3]], axis=1)[:, None]
        T_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    elif set_type == 'LasHeR':
        RGB_img_list = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".png")])
        T_img_list = sorted(
            [seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".png")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')


    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def get_parameters(script_name, yaml_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(script_name))
    params = param_module.parameters(yaml_name)
    return params


def create_tracker(script_name, params, dataset_name):
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(script_name))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params)
    return tracker


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, debug=0, script_name='prompt'):
    seq_txt = seq_name
    # save_name = '{}_{}'.format(script_name, yaml_name)
    save_name = 'test3'
    # save_name = 'w/o_STCF'
    save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = get_parameters(script_name, yaml_name)
    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_
    mmtrack = create_tracker(script_name, params, 'lasher')
    tracker = MoSSTrack_RGBT(tracker=mmtrack)



    seq_path = seq_home + '/' + seq_name
    # seq_path = seq_home + '/' + "large"
    print('——————————Process sequence: ' + seq_name + '——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            region = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()

    np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class MoSSTrack_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        # pred_score = outputs['best_score']
        return pred_bbox


parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
parser.add_argument('--script_name', type=str, default='mosstrack',
                    help='Name of tracking method(ostrack, prompt, ftuning).')
parser.add_argument('--yaml_name', type=str, default='mosstrack_256_full',
                    help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
parser.add_argument('--dataset_name', type=str, default='LasHeR',
                    help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
parser.add_argument('--threads', default=3, type=int, help='Number of threads')
parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
parser.add_argument('--video', default='', type=str, help='specific video name')
args = parser.parse_args()

def main():
    yaml_name = args.yaml_name
    dataset_name = args.dataset_name

    # path initialization
    if dataset_name == 'GTOT':
        seq_home = '/vcl/2025liuyisong/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/vcl/2025liuyisong/RGB_T234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/vcl/tracking_dataset/RGB-T/LasHeR//TestingSet/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/vcl/2025liuyisong/VTUAV/test/short-term'
        with open(join(seq_home, 'VTUAV-ST.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/vcl/2025liuyisong/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    start = time.time()

    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.script_name)
                         for s in seq_list]

        multiprocessing.set_start_method('spawn', force=True)

        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)

    else:
        seq_list_ = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.script_name)
                         for s in seq_list_]

        for seqlist in sequence_list:
            run_sequence(*seqlist)

    print(f"Totally cost {time.time() - start} seconds!")

if __name__ == '__main__':
    main()
