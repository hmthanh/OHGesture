import numpy as np
import h5py
import math
import torch
import torch.nn.functional as F

# [sys.path.append(i) for i in ['./WavLM']]
from wavlm.wavlm_embedding import WavLM, WavLMConfig


def wavlm_init(args, device=torch.device('cuda:0')):
    # load the pre-trained checkpoints
    checkpoint = torch.load(args.wavlm_model_path, map_location=torch.device('cpu'), weights_only=True)
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
        if args.remove_if_exist and os.path.exists(self.h5_dataset):
            os.remove(self.h5_dataset)
        with h5py.File(self.h5_dataset, 'a') as h5:
            print("Dataset h5 before processing: ", len(h5.keys()))

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
                gesture = item['gesture'][:]
                audio_raw = item['speech'][:]
                # mfcc_raw = item['mfcc_raw'][:]
                emotion_raw = item['emotion'][:]
                embedding_raw = item['text'][:]

                self.sample_clip_to_h5_dataset(key, gesture, audio_raw, emotion_raw, embedding_raw)

        with h5py.File(self.h5_dataset, 'r') as h5:
            print("Dataset h5 after processing: ", len(h5.keys()))
            self.n_out_samples = len(h5.keys())

    def sample_read(self):

        with h5py.File(self.h5_data, 'a') as h5:
            for key in h5.keys():
                item = h5[key]
                gesture = item['gesture'][:]
                emotions = item['emotion'][:]
                speech = item['speech'][:]
                text = item['text'][:]

                print("gesture: ", gesture.shape, "emotion: ", emotions.shape, "speech", speech.shape, "text", text.shape)

                break

    def sample_clip_to_h5_dataset(self, key, gesture, audio_raw, emotion_raw, embedding_raw):
        print("Processing data item : " + key)
        # divide
        gesture_list = []
        audio_raw_list = []
        emotion_list = []
        # mfcc_list = []
        wavlm_list = []
        embedding_list = []

        # mfcc_raw,

        # MIN_LEN = min(len(poses), int(len(audio_raw) * 60 / 16000), len(mfcc_raw))
        # MIN_LEN = min(len(poses), int(len(audio_raw) * 60 / 16000))
        total_frames = min(len(gesture), int(len(audio_raw) * 60 / 16000))
        print("Total frames: ", total_frames)

        num_subdivision = math.floor(
            (total_frames - self.n_poses)
            / self.subdivision_stride)  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = gesture[start_idx:fin_idx]
            # sample_mfcc = mfcc_raw[start_idx:fin_idx]
            # subdivision_start_time = start_idx / self.skeleton_resampling_fps
            # subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(gesture) * len(audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = audio_raw[audio_start:audio_end]
            sample_wavlm = wav2wavlm(args, self.wavmodel, sample_audio)

            # embedding
            embedding = embedding_raw[start_idx:fin_idx]

            gesture_list.append(sample_skeletons)
            # mfcc_list.append(sample_mfcc)
            wavlm_list.append(sample_wavlm)
            audio_raw_list.append(sample_audio)
            emotion_list.append(emotion_raw)
            embedding_list.append(embedding)

        n_items = 0
        for gestures, emotions, speeches, embeddings in zip(gesture_list, emotion_list, wavlm_list, embedding_list):
            gestures_np = np.vstack(gestures).astype(np.float32)
            emotions_np = np.asarray(emotions).astype(np.float32)
            speeches_np = np.asarray(speeches).astype(np.float32)
            text_np = np.asarray(embeddings).astype(np.float32)

            print("text_np", text_np.shape, "speeches_np", speeches_np.shape)

            global_key = f'{n_items}_{key}'

            print("Clips : " + str(n_items))

            with h5py.File(self.h5_dataset, 'r+') as h5:
                g_h5 = h5.create_group(global_key)

                g_h5.create_dataset("gesture", data=gestures_np, dtype=np.float32)
                g_h5.create_dataset("emotion", data=emotions_np, dtype=np.float32)
                g_h5.create_dataset("speech", data=speeches_np, dtype=np.float32)
                g_h5.create_dataset("text", data=text_np, dtype=np.float32)

            n_items += 1


if __name__ == '__main__':
    '''
    python data_to_h5dataset.py --config=../main/configs/OHGesture.yml
    '''
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from argparse import ArgumentParser

    parser = ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/OHGesture.yml')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--remove_if_exist', required=False, type=bool, default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)
    args = EasyDict(config)

    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(args.gpu)
    args.device = device

    target = './processed/'
    dataset_path = "./h5dataset"

    data_train_path = os.path.join(target, 'train')
    data_train_h5_name = 'datasets_train.h5'
    data_train = os.path.join(data_train_path, data_train_h5_name)

    dataset_train_h5_name = 'datasets_train.h5'
    dataset_train = os.path.join(dataset_path, dataset_train_h5_name)

    processor = DeepGesturePreprocessor(args, data_train, dataset_train, args.n_poses, args.subdivision_stride, args.motion_resampling_framerate, device)
    processor.run()

    # data_valid_path = os.path.join(target, 'valid')
    # data_valid_h5_name = 'datasets_valid.h5'
    # data_valid = os.path.join(data_valid_path, data_valid_h5_name)
    #
    # dataset_valid_h5_name = 'datasets_valid.h5'
    # dataset_valid = os.path.join(dataset_path, dataset_valid_h5_name)
    #
    # processor = DeepGesturePreprocessor(args, data_valid, dataset_valid, args.n_poses, args.subdivision_stride, args.motion_resampling_framerate, device)
    # processor.run()

    # processor.sample_read()
