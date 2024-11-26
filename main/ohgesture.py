import sys

[sys.path.append(i) for i in ['.', '..', '../model', '../train']]

import pdb
import os
import logging
import torch
import yaml
from torch.utils.data import DataLoader
from pprint import pprint
from easydict import EasyDict

from configs.parse_args import parse_args
from data_loader.deepgesture_dataset import DeepGestureDataset
from utils.model_util import create_gaussian_diffusion
from deepgesture_training_loop import DeepGestureTrainLoop
from ohgesture.deepgesture import DeepGesture

logging.getLogger().setLevel(logging.INFO)


def create_model_and_diffusion(args):
    model = DeepGesture(modeltype='', njoints=1141, nfeats=1,
                        cond_mode='cross_local_attention3_style1', action_emb='tensor',
                        audio_feat=args.audio_feat,
                        arch='trans_enc', latent_dim=256,
                        n_seed=8, cond_mask_prob=0.1)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def main(args, device):
    # ~~~~~~~~~~~~~~~ Train ~~~~~~~~~~~~~~~
    train_dataset = DeepGestureDataset(args.train_h5,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    # ~~~~~~~~~~~~~~~ Valid ~~~~~~~~~~~~~~~
    # val_dataset = DeepGestureDataset(args.valid_h5,
    #                              n_poses=args.n_poses,
    #                              subdivision_stride=args.subdivision_stride,
    #                              pose_resampling_fps=args.motion_resampling_framerate)
    # test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
    #                          shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)

    # logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))
    logging.info('len of train loader: {}'.format(len(train_loader)))

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model, diffusion = create_model_and_diffusion(args)

    model.to(device)
    train_loop = DeepGestureTrainLoop(args, model, diffusion, device, data=train_loader)
    train_loop.run_loop()


if __name__ == '__main__':
    '''
    cd mydiffusion_zeggs/
    python ohgesture.py --config=./configs/OHGesture.yml --gpu mps
    '''

    args = parse_args()
    device = torch.device(args.gpu)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    main(config, device)
