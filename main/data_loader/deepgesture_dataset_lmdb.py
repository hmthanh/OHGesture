import logging
import pdb
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import lmdb


class DeepGestureDatasetLMDB(Dataset):
    def __init__(self, lmdb_path, n_poses, subdivision_stride, pose_resampling_fps):
        """
        Args:
            lmdb_path (str): Path to the LMDB file.
            n_poses (int): Number of poses in a gesture sequence.
            subdivision_stride (int): Stride for subdivisions.
            pose_resampling_fps (int): FPS for resampling the skeleton poses.
        """
        self.lmdb_path = lmdb_path
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        # Open the LMDB environment and determine the length of the dataset
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.dataset_keys = pickle.loads(txn.get("keys".encode()))
            print("Total dataset samples:", len(self.dataset_keys))

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, index):
        """
        Fetch a data sample by its index.
        """
        key = self.dataset_keys[index].encode()
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))

        gesture = data["gesture"]
        emotion = data["emotion"]
        speech = data["speech"]
        text = data["text"]

        return gesture, emotion, speech, text


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        super(RandomSampler, self).__init__()
        self.min_id = min_id
        self.max_id = max_id

    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


if __name__ == '__main__':
    """
    cd main
    python data_loader/deepgesture_dataset.py --config=./configs/OHGesture.yml --gpu mps
    python data_loader/deepgesture_dataset.py --config=./configs/OHGesture.yml --gpu cuda:0
    """

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

    train_dataset = DeepGestureDatasetLMDB(args.train_h5,
                                           n_poses=args.n_poses,
                                           subdivision_stride=args.subdivision_stride,
                                           pose_resampling_fps=args.motion_resampling_framerate)
    # val_dataset = DeepGestureDataset(args.valid_h5,
    #                                    n_poses=args.n_poses,
    #                                    subdivision_stride=args.subdivision_stride,
    #                                    pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print("Total train loader: ", len(train_loader))
    for batch_i, batch in enumerate(train_loader, 0):
        gesture, emotion, speech, embedding = batch
        # print(batch_i)
        # pdb.set_trace()
        print("gesture", gesture.shape)  # gesture torch.Size([1152, 88, 1141])
        print("emotion", emotion.shape)  # emotion torch.Size([1152, 6])
        print("speech", speech.shape)  # speech torch.Size([1152, 88, 1024])
        print("embedding", embedding.shape)  # embedding torch.Size([1152, 88, 300])
