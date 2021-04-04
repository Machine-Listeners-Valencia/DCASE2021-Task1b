import h5py
import numpy as np
from tqdm import tqdm
import os


def normalize_spectrogram(path2train, path2val, input_name, rep='mel'):
    hf_train = h5py.File(path2train, 'r')
    x_train = hf_train['features'][:]
    hf_train.close()

    hf_val = h5py.File(path2val, 'r')
    x_val = hf_val['features'][:]
    hf_val.close()

    ch_0_mean = np.zeros(64)
    ch_1_mean = np.zeros(64)
    ch_2_mean = np.zeros(64)
    ch_0_std = np.zeros(64)
    ch_1_std = np.zeros(64)
    ch_2_std = np.zeros(64)

    for j in tqdm(range(0, 64)):

        points0 = []
        points1 = []
        points2 = []
        for jj in tqdm(range(0, x_train.shape[0])):
            points0 = np.concatenate((points0, x_train[jj, j, :, 0]))
            points1 = np.concatenate((points1, x_train[jj, j, :, 1]))
            points2 = np.concatenate((points2, x_train[jj, j, :, 2]))
        for jj in tqdm(range(0, x_val.shape[0])):
            points0 = np.concatenate((points0, x_val[jj, j, :, 0]))
            points1 = np.concatenate((points1, x_val[jj, j, :, 1]))
            points2 = np.concatenate((points2, x_val[jj, j, :, 2]))

        ch_0_mean[j] = np.mean(points0)
        ch_0_std[j] = np.std(points0)

        ch_1_mean[j] = np.mean(points1)
        ch_1_std[j] = np.std(points1)

        ch_2_mean[j] = np.mean(points2)
        ch_2_std[j] = np.std(points2)

    hf = h5py.File('../data/audiovisual/audio_spectrograms/{}_setup/'
                   'scaler_{}_{}.h5'.format(input_name, rep, input_name), 'w')
    hf.create_dataset('channel_0_mean', data=ch_0_mean)
    hf.create_dataset('channel_1_mean', data=ch_1_mean)
    hf.create_dataset('channel_2_mean', data=ch_2_mean)
    hf.create_dataset('channel_0_std', data=ch_0_std)
    hf.create_dataset('channel_1_std', data=ch_1_std)
    hf.create_dataset('channel_2_std', data=ch_2_std)
    hf.close()


def apply_normalize_spectrogram(path2train, path2val, path2normalizer, input_name, rep='mel'):
    train_file = h5py.File(path2train, 'r')
    train_features = train_file['features'][:]
    train_labels = train_file['labels'][:]
    train_file.close()

    val_file = h5py.File(path2val, 'r')
    val_features = val_file['features'][:]
    val_labels = val_file['labels'][:]
    val_file.close()

    normalize_file = h5py.File(path2normalizer, 'r')
    ch0_mean = normalize_file['channel_0_mean'][:]
    ch1_mean = normalize_file['channel_1_mean'][:]
    ch2_mean = normalize_file['channel_2_mean'][:]
    ch0_std = normalize_file['channel_0_std'][:]
    ch1_std = normalize_file['channel_1_std'][:]
    ch2_std = normalize_file['channel_2_std'][:]

    for j in tqdm(range(0, train_features.shape[0])):
        for jj in range(0, train_features.shape[1]):
            train_features[j, jj, :, 0] = (train_features[j, jj, :, 0] - ch0_mean[jj]) / ch0_std[jj]
            train_features[j, jj, :, 1] = (train_features[j, jj, :, 1] - ch1_mean[jj]) / ch1_std[jj]
            train_features[j, jj, :, 2] = (train_features[j, jj, :, 2] - ch2_mean[jj]) / ch2_std[jj]

    for j in tqdm(range(0, val_features.shape[0])):
        for jj in range(0, val_features.shape[1]):
            val_features[j, jj, :, 0] = (val_features[j, jj, :, 0] - ch0_mean[jj]) / ch0_std[jj]
            val_features[j, jj, :, 1] = (val_features[j, jj, :, 1] - ch1_mean[jj]) / ch1_std[jj]
            val_features[j, jj, :, 2] = (val_features[j, jj, :, 2] - ch2_mean[jj]) / ch2_std[jj]

    if os.path.isdir('../data/audiovisual/audio_normalized_spectrograms/'
                     '{}_setup/'.format(input_name)) is False:
        os.mkdir('../data/audiovisual/audio_normalized_spectrograms/{}_setup/'.format(input_name))

    if rep == 'gammatone':
        hf = h5py.File('../data/audiovisual/audio_normalized_spectrograms/{}_setup/'
                       'training_setup_{}_{}.h5'.format(input_name, input_name, rep), 'w')
    else:
        hf = h5py.File('../data/audiovisual/audio_normalized_spectrograms/{}_setup/'
                       'training_setup_{}.h5'.format(input_name, input_name), 'w')

    hf.create_dataset("features", data=train_features)
    hf.create_dataset("labels", data=train_labels)
    hf.close()

    if rep == 'gammatone':
        hf = h5py.File('../data/audiovisual/audio_normalized_spectrograms/{}_setup/'
                       'val_setup_{}_{}.h5'.format(input_name, input_name, rep), 'w')
    else:
        hf = h5py.File('../data/audiovisual/audio_normalized_spectrograms/{}_setup/'
                       'val_setup_{}.h5'.format(input_name, input_name), 'w')
    hf.create_dataset("features", data=val_features)
    hf.create_dataset("labels", data=val_labels)
    hf.close()


if __name__ == '__main__':
    # path2train = '/home/javi/repos/DCASE2021-Task1b/data/audiovisual' \
    #              '/audio_spectrograms/lrd_setup/training_setup_lrd_gammatone.h5'
    # path2val = '/home/javi/repos/DCASE2021-Task1b/data/audiovisual' \
    #            '/audio_spectrograms/lrd_setup/val_setup_lrd_gammatone.h5'
    #
    # normalize_spectrogram(path2train, path2val)

    path2train = '../data/audiovisual/audio_spectrograms/lrd_setup/training_setup_lrd.h5'
    path2val = '../data/audiovisual/audio_spectrograms/lrd_setup/val_setup_lrd.h5'
    path2normalizer = '../data/audiovisual/audio_spectrograms/lrd_setup/scaler_mel_lrd.h5'
    input_name = 'lrd'

    apply_normalize_spectrogram(path2train, path2val, path2normalizer, input_name, rep='mel')