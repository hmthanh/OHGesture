import logging
import pdb
import lmdb as lmdb
import torch
from torch.utils.data import Dataset
import pyarrow
import sys
import os
import numpy as np
[sys.path.append(i) for i in ['.', '..']]
import h5py

class DeepGestureDataset(Dataset):
    def __init__(self, h5_dataset_file, n_poses, subdivision_stride, pose_resampling_fps):
        self.h5_dataset_file = h5_dataset_file
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        self.speeches = []
        self.gestures = []
        self.emotions = []

        with h5py.File(self.h5_dataset_file, 'r') as h5:
            self.n_samples = len(h5.keys())
            self.dataset_keys = list(h5.keys())
            print("Total dataset samples : ", self.n_samples)

            for key in self.dataset_keys:
                self.gestures.append(h5[key]["gesture"][:])
                self.emotions.append(h5[key]["emotion"][:])
                self.speeches.append(h5[key]["speech"][:])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gesture = self.gestures[idx]
        emotion = self.emotions[idx]
        speech = self.speeches[idx]

        return gesture, emotion, speech
        # with self.lmdb_env.begin(write=False) as txn:
        #     key = '{:010}'.format(idx).encode('ascii')
        #     sample = txn.get(key)
        #
        #     # sample = pyarrow.deserialize(sample)
        #     sample_value = pyarrow.ipc.deserialize_pandas(sample)
        #     poses =  sample_value["poses"][0]
        #     codes = sample_value["codes"][0]
        #     wavlm = sample_value["wavlm"][0]
        #
        #     poses = np.vstack(poses).astype(np.float64)
        #     pose_seq = np.asarray(poses, dtype=np.float64)
        #
        #     styles = np.asarray(codes, dtype=np.float32)
        #     wavlm = np.vstack(wavlm).astype(np.float64)
        #     wavlm = np.asarray(wavlm, dtype=np.float64)
        #     # pose_seq, audio, styles, mfcc, wavlm, aux_info = sample
        #     # pose_seq, styles, wavlm = sample
        #
        # # # normalize
        # # std = np.clip(self.data_std, a_min=0.01, a_max=None)
        # # pose_seq = (pose_seq - self.data_mean) / std
        #
        # # to tensors
        # pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        # styles = torch.from_numpy(styles).float()
        # # audio = torch.from_numpy(audio).float()
        # # mfcc = torch.from_numpy(mfcc).float()
        # wavlm = torch.from_numpy(wavlm).float()
        #
        # # return pose_seq, aux_info, styles, audio, mfcc, wavlm
        # return pose_seq, styles, wavlm


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
        gesture, emotion, speech = batch
        print(batch_i)
        # pdb.set_trace()
        print(gesture.shape) # torch.Size([128, 88, 1141])
        print(emotion.shape) # torch.Size([128, 6])
        print(speech.shape) # torch.Size([128, 88, 1024])
