import logging
import pdb

import sys
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def custom_collate(batch):
    """
    Custom collate function for DeepGestureDataset to handle variable-length sequences.

    Args:
        batch (list): A list of tuples from the dataset

    Returns:
        tuple: Padded and batched tensors
    """
    # Separate the batch components
    gestures, emotions, speeches, texts = zip(*batch)

    # Convert to tensors
    gestures = [torch.tensor(gesture, dtype=torch.float32) for gesture in gestures]
    emotions = [torch.tensor(emotion, dtype=torch.float32) for emotion in emotions]
    speeches = [torch.tensor(speech, dtype=torch.float32) for speech in speeches]
    texts = [torch.tensor(text, dtype=torch.long) for text in texts]

    # Pad sequences
    padded_gestures = pad_sequence(gestures, batch_first=True)
    padded_emotions = pad_sequence(emotions, batch_first=True)
    padded_speeches = pad_sequence(speeches, batch_first=True)

    # Create a mask for variable-length sequences (optional but recommended)
    gesture_lengths = torch.tensor([len(gesture) for gesture in gestures])
    emotion_lengths = torch.tensor([len(emotion) for emotion in emotions])
    speech_lengths = torch.tensor([len(speech) for speech in speeches])

    # Pad texts (assuming texts might be sequences of tokens)
    padded_texts = pad_sequence(texts, batch_first=True)
    text_lengths = torch.tensor([len(text) for text in texts])

    return (
        padded_gestures,
        padded_emotions,
        padded_speeches,
        padded_texts
    )


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

    train_dataset = DeepGestureDataset(args.train_h5,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate)
    # val_dataset = DeepGestureDataset(args.valid_h5,
    #                                    n_poses=args.n_poses,
    #                                    subdivision_stride=args.subdivision_stride,
    #                                    pose_resampling_fps=args.motion_resampling_framerate)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                           shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.loader_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )

    print("Total train loader: ", len(train_loader))
    for batch_i, batch in enumerate(train_loader, 0):
        gesture, emotion, speech, embedding = batch
        # print(batch_i)
        # pdb.set_trace()
        print("gesture", gesture.shape)  # gesture torch.Size([1152, 88, 1141])
        print("emotion", emotion.shape)  # emotion torch.Size([1152, 6])
        print("speech", speech.shape)  # speech torch.Size([1152, 88, 1024])
        print("embedding", embedding.shape)  # embedding torch.Size([1152, 88, 300])
        break

