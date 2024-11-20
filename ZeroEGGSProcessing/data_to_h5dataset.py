import os
import glob
import subprocess
import numpy as np
import lmdb
from mfcc import MFCC
import soundfile as sf
import sys
import h5py
import math
from wavlm.wavlm_embedding import WavLM, WavLMConfig
import torch
import torch.nn.functional as F
import platform

# import sys
# [sys.path.append(i) for i in ['./WavLM']]

def wavlm_init(args, device=torch.device('cuda:0')):
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    # wavlm_model_path = '../../../My/process/WavLM-Base+.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'), weights_only=True)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(args, model, wav_input_16khz):
    with torch.no_grad():
        wav_input_16khz = np.copy(wav_input_16khz)
        wav_input_16khz = torch.from_numpy(wav_input_16khz).float()
        wav_input_16khz = wav_input_16khz.to(args.device).unsqueeze(0)

        rep = model.extract_features(wav_input_16khz)[0]
        rep = F.interpolate(rep.transpose(1, 2), size=88, align_corners=True, mode='linear').transpose(1, 2)

        return rep.squeeze().cpu().detach().data.cpu().numpy()


class DeepGesturePreprocessor:
    def __init__(self, args, h5_data, h5_dataset, n_poses, subdivision_stride, pose_resampling_fps, device):
        self.h5_data = h5_data
        with h5py.File(self.h5_data, 'r') as hdf:
            self.h5_data_keys = list(hdf.keys())

            if len(self.h5_data_keys) <= 0:
                print("Dataset is empty, please processing ZEGGS first.")

        self.h5_dataset = h5_dataset
        # if os.path.exists(self.h5_dataset):
        #     os.remove(self.h5_dataset)
        with h5py.File(self.h5_dataset, 'a') as h5:
            print("Output h5 dataset before processing: ", len(h5.keys()))

        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.device = device
        self.wavmodel = wavlm_init(args, device)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)
        self.n_out_samples = 0


    def run(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for key in self.h5_data_keys:
            # print(key)
            with h5py.File(self.h5_data, 'r') as hdf:
                item = hdf[key]

                # Access datasets
                poses = item['poses'][:]
                audio_raw = item['audio_raw'][:]
                mfcc_raw = item['mfcc_raw'][:]
                style_raw = item['style_raw'][:]

                self.sample_clip_to_h5_dataset(key, poses, audio_raw, mfcc_raw, style_raw)

        with h5py.File(self.h5_dataset, 'r') as h5:
            print("Output h5 dataset after processing: ", len(h5.keys()))
            self.n_out_samples = len(h5.keys())

    def sample(self):
        poses_list = []
        audio_list = []
        mfcc_list = []
        style_list = []

        print("Processing dataset: " + self.h5_data)
        with h5py.File(self.h5_data, 'r') as hdf:
            for key in hdf.keys():
                item = hdf[key]

                # Access datasets
                poses = item['poses'][:]
                audio_raw = item['audio_raw'][:]
                mfcc_raw = item['mfcc_raw'][:]
                style_raw = item['style_raw'][:]

                poses_list.append(poses)
                audio_list.append(audio_raw)
                mfcc_list.append(mfcc_raw)
                style_list.append(style_raw)
                print("Processing dataset: " + self.h5_data)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(f"poses: {len(poses_list)}")
        print(f"audio_raw: {len(audio_list)}")
        print(f"mfcc_raw: {len(mfcc_list)}")
        print(f"style_raw: {len(style_list)}")

    def sample_clip_to_h5_dataset(self, key, poses, audio_raw, mfcc_raw, style_raw):
        print("Processing data item : " + key)
        # divide
        # aux_info = []
        gesture_list = []
        audio_raw_list = []
        emotion_list = []
        mfcc_list = []
        wavlm_list = []

        MIN_LEN = min(len(poses), int(len(audio_raw) * 60 / 16000), len(mfcc_raw))

        num_subdivision = math.floor(
            (MIN_LEN - self.n_poses)
            / self.subdivision_stride)  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = poses[start_idx:fin_idx]
            sample_mfcc = mfcc_raw[start_idx:fin_idx]
            # subdivision_start_time = start_idx / self.skeleton_resampling_fps
            # subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(poses) * len(audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = audio_raw[audio_start:audio_end]
            sample_wavlm = wav2wavlm(args, self.wavmodel, sample_audio)

            # motion_info = {'vid': vid,
            #                'start_frame_no': start_idx,
            #                'end_frame_no': fin_idx,
            #                'start_time': subdivision_start_time,
            #                'end_time': subdivision_end_time}

            gesture_list.append(sample_skeletons)
            mfcc_list.append(sample_mfcc)
            wavlm_list.append(sample_wavlm)
            audio_raw_list.append(sample_audio)
            emotion_list.append(style_raw)
            # aux_info.append(motion_info)

        n_items = 0
        for gestures, emotions, speeches in zip(gesture_list, emotion_list, wavlm_list):
            gestures_np = np.vstack(gestures).astype(np.float32)
            emotions_np = np.asarray(emotions).astype(np.float32)
            speeches_np = np.asarray(speeches).astype(np.float32)

            global_key = f'{n_items}_{key}'

            print("Clips : " + str(n_items))

            with h5py.File(self.h5_dataset, 'r+') as h5:
                g_h5 = h5.create_group(global_key)

                g_h5.create_dataset("gesture", data=gestures_np, dtype=np.float32)
                g_h5.create_dataset("emotion", data=emotions_np, dtype=np.float32)
                g_h5.create_dataset("wavlm", data=speeches_np, dtype=np.float32)

            n_items += 1


if __name__ == '__main__':
    '''
    python data_to_h5dataset.py --config=../mydiffusion_zeggs/configs/OHGesture.yml
    '''
    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # if platform.system() == 'Darwin' and torch.backends.mps.is_available():
    #     config['device'] = torch.device('mps')  # Use MPS for MacBooks with Apple Silicon
    # else:
    #     config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)
    args = EasyDict(config)
    device = torch.device(args.gpu)
    args.device = device

    target = './processed/'
    dataset_path = "./h5dataset"

    data_train_path = os.path.join(target, 'train')
    data_train_h5_name = 'datasets_train.h5'
    data_train = os.path.join(data_train_path, data_train_h5_name)

    dataset_train_h5_name = 'datasets_train.h5'
    dataset_train = os.path.join(dataset_path, dataset_train_h5_name)

    # processor = DeepGesturePreprocessor(args, data_train, dataset_train, args.n_poses, args.subdivision_stride, args.motion_resampling_framerate, device)
    # processor.run()

    data_valid_path = os.path.join(target, 'valid')
    data_valid_h5_name = 'datasets_valid.h5'
    data_valid = os.path.join(data_valid_path, data_valid_h5_name)

    dataset_valid_h5_name = 'datasets_valid.h5'
    dataset_valid = os.path.join(dataset_path, dataset_valid_h5_name)

    processor = DeepGesturePreprocessor(args, data_valid, dataset_valid, args.n_poses, args.subdivision_stride, args.motion_resampling_framerate, device)
    processor.run()