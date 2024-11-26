import logging
import pdb
import torch
from torch.utils.data import Dataset
import sys
import os
import h5py


class DeepGestureDataset(Dataset):
    def __init__(self, h5_dataset_file, n_poses, subdivision_stride, pose_resampling_fps):
        self.h5_dataset_file = h5_dataset_file
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        with h5py.File(self.h5_dataset_file, 'r') as h5:
            self.dataset_keys = list(h5.keys())
            print("Total dataset samples : ", len(self.dataset_keys))

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, index):
        with h5py.File(self.h5_dataset_file, 'r') as h5:
            gesture = h5[self.dataset_keys[index]]["gesture"][:]
            emotion = h5[self.dataset_keys[index]]["emotion"][:]
            speech = h5[self.dataset_keys[index]]["speech"][:]
            text = h5[self.dataset_keys[index]]["text"][:]

            return gesture, emotion, speech, text


if __name__ == '__main__':
    '''
    cd main/mydiffusion_zeggs
    python data_loader/deepgesture_dataset.py --config=./configs/OHGesture.yml --gpu mps
    python data_loader/deepgesture_dataset.py --config=./configs/OHGesture.yml --gpu cuda:0
    '''

    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    args = parse_args()

    device = torch.device(args.gpu)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    args = EasyDict(config)

    train_dataset = DeepGestureDataset(args.train_h5,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate)
    # val_dataset = DeepGestureDataset(args.valid_h5,
    #                                    n_poses=args.n_poses,
    #                                    subdivision_stride=args.subdivision_stride,
    #                                    pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)
    #
    print(len(train_loader))
    for batch_i, batch in enumerate(train_loader, 0):
        # target_vec, aux, style, audio, mfcc, wavlm = batch     # [128, 88, 1141], -,  [128, 6], [128, 70400], [128, 88, 13]
        gesture, emotion, speech, embedding = batch
        print(batch_i)
        # pdb.set_trace()
        print(gesture.shape)  # torch.Size([128, 88, 1141])
        print(emotion.shape)  # torch.Size([128, 6])
        print(speech.shape)  # torch.Size([128, 88, 1024])
        print(embedding.shape)  # torch.Size([128, 88, 1024])
