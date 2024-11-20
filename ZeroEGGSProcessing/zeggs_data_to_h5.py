import os
import glob
import subprocess
import numpy as np
import lmdb
from mfcc import MFCC
import soundfile as sf
import sys
import h5py

from process_zeggs_bvh import preprocess_animation, pose2bvh


[sys.path.append(i) for i in ['.', '..', '../process']]

style2onehot = {
    'Happy': [1, 0, 0, 0, 0, 0],
    'Sad': [0, 1, 0, 0, 0, 0],
    'Neutral': [0, 0, 1, 0, 0, 0],
    'Old': [0, 0, 0, 1, 0, 0],
    'Angry': [0, 0, 0, 0, 1, 0],
    'Relaxed': [0, 0, 0, 0, 0, 1],
}


def make_lmdb_gesture_dataset(root_path):
    def make_lmdb_gesture_subdataset(base_path, lmdb_subname):
        gesture_path = os.path.join(base_path, 'gesture_npz')
        audio_path = os.path.join(base_path, 'normalize_audio_npz')
        mfcc_path = os.path.join(base_path, 'mfcc')
        out_path = os.path.join(base_path, lmdb_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # map_size = 1024 * 200  # in MB
        # map_size <<= 20  # in B
        # dataset_idx = 0
        # print("map_size", map_size)
        map_size = 10 * 1024 * 1024 * 1024

        lmdb_path = os.path.normpath(os.path.join(os.path.normpath(out_path), lmdb_subname))
        db = [lmdb.open(lmdb_path, map_size=map_size)]

        # delete existing files
        for i in range(1):
            with db[i].begin(write=True) as txn:
                txn.drop(db[i].open_db())

        all_poses = []
        bvh_files = sorted(glob.glob(gesture_path + "/*.npz"))
        v_i = 0

        for _, bvh_file in enumerate(bvh_files):
            name = os.path.split(bvh_file)[1][:-4]
            if name.split('_')[1] in style2onehot:
                style = style2onehot[name.split('_')[1]]
            else:
                continue

            print('process: ' + name)

            poses = np.load(bvh_file)['gesture']
            audio_raw = np.load(os.path.join(audio_path, name + '.npz'))['wav']
            mfcc_raw = np.load(os.path.join(mfcc_path, name + '.npz'))['mfcc']

            # process
            data_mean = np.load(os.path.join(root_path, 'mean.npz'))['mean']
            data_std = np.load(os.path.join(root_path, 'std.npz'))['std']
            data_mean = np.array(data_mean).squeeze()
            data_std = np.array(data_std).squeeze()
            std = np.clip(data_std, a_min=0.01, a_max=None)
            poses = (poses - data_mean) / std

            poses = np.asarray(poses)
            # clips = [{'vid': name, 'clips': []}]  # train and test
            # clips[dataset_idx]['clips'].append(
            #     {  # 'words': word_list,
            #         'poses': poses,
            #         'audio_raw': audio_raw,
            #         'mfcc_raw': mfcc_raw,  # for debug
            #         'style_raw': np.array(style)  # for debug
            #     })
            # df = pd.DataFrame([{
            #     'vid': name,
            #     'clips': [{
            #         'poses': poses.tolist(),
            #         'audio_raw': audio_raw.tolist(),
            #         'mfcc_raw': mfcc_raw.tolist(),  # for debug
            #         'style_raw': np.array(style).tolist()  # for debug
            #     }]
            # }])
            #
            # # write to db
            # for i in range(1):
            #     with db[i].begin(write=True) as txn:
            #         # if len(clips[i]['clips']) > 0:
            #         k = '{:010}'.format(v_i).encode('ascii')
            #         # v = pa.serialize_pandas(df)
            #         v = pa.ipc.serialize_pandas(df).to_pybytes()
            #         # v = pyarrow.serialize(clips[i]).to_buffer()
            #         txn.put(k, v)
            #
            # all_poses.append(poses)
            v_i += 1

        print('total length of dataset: ' + str(v_i))

        # close db
        for i in range(1):
            db[i].sync()
            db[i].close()

    train_path = os.path.join(root_path, 'train')
    lmdb_name = 'train_lmdb'
    make_lmdb_gesture_subdataset(train_path, lmdb_name)
    test_path = os.path.join(root_path, 'valid')
    lmdb_name = 'valid_lmdb'
    make_lmdb_gesture_subdataset(test_path, lmdb_name)


def make_h5_gesture_dataset(root_path):
    def make_lmdb_gesture_subdataset(base_path, h5_name):
        h5_dataset_path = os.path.join(base_path, h5_name)

        gesture_path = os.path.join(base_path, 'gesture_npz')
        audio_path = os.path.join(base_path, 'normalize_audio_npz')
        mfcc_path = os.path.join(base_path, 'mfcc')

        bvh_files = sorted(glob.glob(gesture_path + "/*.npz"))
        total_bvh_files = len(bvh_files)
        v_i = 0
        print(f"Processing total {total_bvh_files} bvh files")

        with h5py.File(h5_dataset_path, 'w') as hdf:
            print("Created ", h5_dataset_path)

            for _, bvh_file in enumerate(bvh_files):
                name = os.path.split(bvh_file)[1][:-4]
                print('processing: ' + name)
                if name.split('_')[1] in style2onehot:
                    style = style2onehot[name.split('_')[1]]
                else:
                    continue

                # load data
                poses = np.load(bvh_file)['gesture']
                audio_raw = np.load(os.path.join(audio_path, name + '.npz'))['wav']
                mfcc_raw = np.load(os.path.join(mfcc_path, name + '.npz'))['mfcc']

                # process mean and std
                data_mean = np.load(os.path.join(root_path, 'mean.npz'))['mean']
                data_std = np.load(os.path.join(root_path, 'std.npz'))['std']
                data_mean = np.array(data_mean).squeeze()
                data_std = np.array(data_std).squeeze()
                std = np.clip(data_std, a_min=0.01, a_max=None)
                poses = (poses - data_mean) / std

                poses = np.asarray(poses)

                # ~~~~~~~~~~~~~ Write Data ~~~~~~~~~~~~~
                g_data = hdf.create_group(name)

                g_data.create_dataset('poses', data=poses)
                g_data.create_dataset('audio_raw', data=audio_raw)
                g_data.create_dataset('mfcc_raw', data=mfcc_raw)
                g_data.create_dataset('style_raw', data=np.array(style))

                v_i += 1

            print('total length of dataset: ' + str(v_i))

        assert total_bvh_files == v_i

    train_path = os.path.join(root_path, 'train')
    train_h5_name = 'datasets_train.h5'
    make_lmdb_gesture_subdataset(train_path, train_h5_name)

    test_path = os.path.join(root_path, 'valid')
    valid_h5_name = 'datasets_valid.h5'
    make_lmdb_gesture_subdataset(test_path, valid_h5_name)

def make_zeggs_dataset(source_path, target):
    if not os.path.exists(target):
        os.mkdir(target)

    def make_zeggs_subdataset(source_path, target, all_poses):
        if not os.path.exists(target):
            os.mkdir(target)

        target_audio_path = os.path.join(target, 'normalize_audio')
        target_audionpz_path = os.path.join(target, 'normalize_audio_npz')
        target_gesture_path = os.path.join(target, 'gesture_npz')
        target_mfcc_path = os.path.join(target, 'mfcc')

        if not os.path.exists(target_audio_path):
            os.mkdir(target_audio_path)
        if not os.path.exists(target_mfcc_path):
            os.mkdir(target_mfcc_path)
        if not os.path.exists(target_audionpz_path):
            os.mkdir(target_audionpz_path)
        if not os.path.exists(target_gesture_path):
            os.mkdir(target_gesture_path)

        wav_files = sorted(glob.glob(source_path + "/*.wav"))
        print("Processing total " + str(len(wav_files)) + " wav files")

        for _, wav_file in enumerate(wav_files):
            name = os.path.split(wav_file)[1][:-4]
            print(name)

            # audio
            print('normalize audio: ' + name + '.wav')
            normalize_wav_path = os.path.join(target_audio_path, name + '.wav')
            cmd = ['ffmpeg-normalize', wav_file, '-o', normalize_wav_path, '-ar', '16000']
            subprocess.call(cmd)

            print('extract MFCC...')
            obj = MFCC(frate=20)
            # wav, fs = librosa.load(normalize_wav_path, sr=16000)
            wav, fs = sf.read(normalize_wav_path)
            mfcc = obj.sig2s2mfc_energy(wav, None)
            print(mfcc[:, :-2].shape)  # -1 -> -2      # (502, 13)
            np.savez_compressed(os.path.join(target_mfcc_path, name + '.npz'), mfcc=mfcc[:, :-2])
            np.savez_compressed(os.path.join(target_audionpz_path, name + '.npz'), wav=wav)

            # bvh
            print('extract gesture...')
            bvh_file = os.path.join(source_path, name + '.bvh')
            pose, parents, dt, order, njoints = preprocess_animation(bvh_file, fps=20)
            print(pose.shape)
            np.savez_compressed(os.path.join(target_gesture_path, name + '.npz'), gesture=pose)
            all_poses.append(pose)

        return all_poses

    source_path_train = os.path.join(source_path, 'train')
    target_train = os.path.join(target, 'train')
    all_poses = []
    all_poses = make_zeggs_subdataset(source_path_train, target_train, all_poses)
    source_path_test = os.path.join(source_path, 'valid')
    target_test = os.path.join(target, 'valid')
    all_poses = make_zeggs_subdataset(source_path_test, target_test, all_poses)

    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float32)
    pose_std = np.std(all_poses, axis=0, dtype=np.float32)
    np.savez_compressed(os.path.join(target, 'mean.npz'), mean=pose_mean)
    np.savez_compressed(os.path.join(target, 'std.npz'), std=pose_std)


if __name__ == '__main__':
    '''
    python zeggs_data_to_lmdb.py
    '''
    source_path = './data/'
    target = './processed/'
    # make_zeggs_dataset(source_path, target)
    make_h5_gesture_dataset(target)

    # def sample_read_h5_dataset(h5_dataset_path):
    #     poses_list = []
    #     audio_list = []
    #     mfcc_list = []
    #     style_list = []
    #
    #     print("Processing dataset: " + h5_dataset_path)
    #     with h5py.File(h5_dataset_path, 'r') as hdf:
    #         for key in hdf.keys():
    #             item = hdf[key]
    #
    #             # Access datasets
    #             poses = item['poses'][:]
    #             audio_raw = item['audio_raw'][:]
    #             mfcc_raw = item['mfcc_raw'][:]
    #             style_raw = item['style_raw'][:]
    #
    #             poses_list.append(poses)
    #             audio_list.append(audio_raw)
    #             mfcc_list.append(mfcc_raw)
    #             style_list.append(style_raw)
    #             print("Processing dataset: " + h5_dataset_path)
    #
    #     print(f"poses: {len(poses_list)}")
    #     print(f"audio_raw: {len(audio_list)}")
    #     print(f"mfcc_raw: {len(mfcc_list)}")
    #     print(f"style_raw: {len(style_list)}")
    #
    #
    # train_path = os.path.join(target, 'train')
    # train_h5_name = 'datasets_train.h5'
    # sample_read_h5_dataset(os.path.join(train_path, train_h5_name))
    #
    # valid_path = os.path.join(target, 'valid')
    # valid_h5_name = 'datasets_valid.h5'
    # sample_read_h5_dataset(os.path.join(valid_path, valid_h5_name))
