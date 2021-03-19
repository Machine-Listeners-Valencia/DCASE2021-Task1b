import pandas as pd
from tqdm import tqdm
import librosa
import pickle
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

'''
Too hardcoded code
'''


def get_1s_mel_spectrogram(path2csv, n_mels=64, win_len=0.04, hop_len=0.02, mono=True, val=False,
                           folder_name=None):
    dataframe = pd.read_csv(path2csv, sep='\t')

    audio_files = dataframe['filename_audio'].tolist()
    audio_labels = dataframe['scene_label'].tolist()

    for ii in tqdm(range(0, len(audio_files))):
        y, sr = librosa.load('../data/audiovisual/{}'.format(audio_files[ii]),
                             sr=None, mono=mono)

        if mono:
            y = librosa.to_mono(y)
            folder_name = 'mono'
        else:
            folder_name = folder_name

        if mono:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=int(sr / 2),
                                               win_length=int(win_len * sr),
                                               hop_length=int(hop_len * sr))

            S_db = librosa.power_to_db(S)
            S_db = S_db[:, :500]

        elif folder_name == 'lrd':
            Sl = librosa.feature.melspectrogram(y=np.asfortranarray(y[0].copy()), sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_l = librosa.power_to_db(Sl)
            S_db_l = S_db_l[:, :500]
            Sr = librosa.feature.melspectrogram(y=np.asfortranarray(y[1].copy()), sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_r = librosa.power_to_db(Sr)
            S_db_r = S_db_r[:, :500]
            Sd = librosa.feature.melspectrogram(y=np.asfortranarray(y[0].copy()) - np.asfortranarray(y[1].copy()),
                                                sr=sr, n_mels=n_mels, fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_d = librosa.power_to_db(Sd)
            S_db_d = S_db_d[:, :500]

            # TODO: not working
            S_db = np.dstack([S_db_l, S_db_r, S_db_d])

            # 10 clips per audio
            for jj in range(0, 10):
                S_db_aux = S_db[:, jj * 50:(jj + 1) * 50]
                filename_aux = os.path.splitext(os.path.basename(audio_files[ii]))[0]
                if val:
                    with open('../data/audiovisual/audio_spectrograms/'
                              'val_{}/{}_{}.pickle'.format(folder_name, filename_aux, jj), 'wb') as f:
                        pickle.dump([S_db_aux, audio_labels[ii]], f)
                    f.close()
                else:
                    with open('../data/audiovisual/audio_spectrograms/'
                              'train_{}/{}_{}.pickle'.format(folder_name, filename_aux, jj), 'wb') as f:
                        pickle.dump([S_db_aux, audio_labels[ii]], f)

                    f.close()


def compact_audio_files(path2spectrograms, n_mels, n_channels):
    onlyfiles = [f for f in listdir(path2spectrograms) if isfile(join(path2spectrograms, f))]

    if n_channels is not None:
        features = np.zeros((len(onlyfiles), n_mels, 50, n_channels))
    else:
        features = np.zeros((len(onlyfiles), n_mels, 50))
    labels = [None] * len(onlyfiles)

    for i in range(0, len(onlyfiles)):
        with open(path2spectrograms + '/' + onlyfiles[i], 'rb') as f:  # Python 3: open(..., 'rb')
            features_aux, labels_aux = pickle.load(f)
        features[i] = features_aux
        labels[i] = labels_aux

    # Saving the objects:
    # with open('training_setup.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([features, labels], f)
    hf = h5py.File('training_setup_lrd.h5', 'w')
    hf.create_dataset("features", data=features)
    hf.create_dataset("labels", labels)
    hf.close()


if __name__ == '__main__':
    path2csv_train = '../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/evaluation_setup' \
                     '/fold1_train.csv'
    path2csv_evaluate = '../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/evaluation_setup' \
                        '/fold1_evaluate.csv'

    # get_1s_mel_spectrogram(path2csv_evaluate, mono=False, val=True, folder_name='lrd')
    path2spectrograms = '../data/audiovisual/audio_spectrograms/train_lrd'
    n_mels = 64
    compact_audio_files(path2spectrograms, n_mels, 3)
