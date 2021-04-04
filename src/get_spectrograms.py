import pandas as pd
import librosa
import pickle
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from gammatone.gtgram import gtgram

'''
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
https://stackoverflow.com/questions/28656736/using-scikits-labelencoder-correctly-across-multiple-programs
'''


def get_1s_mel_spectrogram(path2csv, n_mels=64, win_len=0.04, hop_len=0.02, mono=True, val=False,
                           folder_name=None):
    dataframe = pd.read_csv(path2csv, sep='\t')

    audio_files = dataframe['filename_audio'].tolist()
    audio_labels = dataframe['scene_label'].tolist()

    if os.path.isdir('../data/audiovisual/audio_spectrograms/val_{}/'.format(folder_name)) is False:
        os.mkdir('../data/audiovisual/audio_spectrograms/val_{}/'.format(folder_name))

    if os.path.isdir('../data/audiovisual/audio_spectrograms/train_{}/'.format(folder_name)) is False:
        os.mkdir('../data/audiovisual/audio_spectrograms/train_{}/'.format(folder_name))

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

            S_db = librosa.power_to_db(np.abs(S), ref=np.max)
            S_db = S_db[:, :500]

        elif folder_name == 'lrd':
            Sl = librosa.feature.melspectrogram(y=np.asfortranarray(y[0].copy()), sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_l = librosa.power_to_db(np.abs(Sl), ref=np.max)
            S_db_l = S_db_l[:, :500]
            Sr = librosa.feature.melspectrogram(y=np.asfortranarray(y[1].copy()), sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_r = librosa.power_to_db(np.abs(Sr), ref=np.max)
            S_db_r = S_db_r[:, :500]
            Sd = librosa.feature.melspectrogram(y=np.asfortranarray(y[0].copy()) - np.asfortranarray(y[1].copy()),
                                                sr=sr, n_mels=n_mels, fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_d = librosa.power_to_db(np.abs(Sd), ref=np.max)
            S_db_d = S_db_d[:, :500]

            S_db = np.dstack([S_db_l, S_db_r, S_db_d])

        elif folder_name == 'hpd':

            y_mono = librosa.to_mono(y)

            y_harmonic, y_percussive = librosa.effects.hpss(y_mono)

            Sh = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_h = librosa.power_to_db(np.abs(Sh), ref=np.max)
            S_db_h = S_db_h[:, :500]

            Sp = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=n_mels,
                                                fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_p = librosa.power_to_db(np.abs(Sp), ref=np.max)
            S_db_p = S_db_p[:, :500]

            Sd = librosa.feature.melspectrogram(y=np.asfortranarray(y[0].copy()) - np.asfortranarray(y[1].copy()),
                                                sr=sr, n_mels=n_mels, fmax=int(sr / 2),
                                                win_length=int(win_len * sr),
                                                hop_length=int(hop_len * sr))
            S_db_d = librosa.power_to_db(np.abs(Sd), ref=np.max)
            S_db_d = S_db_d[:, :500]

            S_db = np.dstack([S_db_h, S_db_p, S_db_d])

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


def get_1s_gammatone_spectrogram(path2csv, n_channels=64, win_len=0.04, hop_len=0.02, mono=True,
                                 val=False, folder_name=None):
    dataframe = pd.read_csv(path2csv, sep='\t')

    audio_files = dataframe['filename_audio'].tolist()
    audio_labels = dataframe['scene_label'].tolist()

    if os.path.isdir('../data/audiovisual/audio_spectrograms/gammatone_val_{}/'.format(folder_name)) is False:
        os.mkdir('../data/audiovisual/audio_spectrograms/gammatone_val_{}/'.format(folder_name))

    if os.path.isdir('../data/audiovisual/audio_spectrograms/gammatone_train_{}/'.format(folder_name)) is False:
        os.mkdir('../data/audiovisual/audio_spectrograms/gammatone_train_{}/'.format(folder_name))

    for ii in tqdm(range(0, len(audio_files))):

        if folder_name == 'lrd':
            for jj in range(0, 10):
                y, sr = librosa.load('../data/audiovisual/{}'.format(audio_files[ii]),
                                     sr=None, mono=mono, offset=float(jj), duration=1)

                y_l = np.asfortranarray(y[0].copy())
                y_r = np.asfortranarray(y[1].copy())
                y_d = y_l - y_r

                S_l = gtgram(y_l, sr, win_len, hop_len, n_channels, 0)
                S_r = gtgram(y_r, sr, win_len, hop_len, n_channels, 0)
                S_d = gtgram(y_d, sr, win_len, hop_len, n_channels, 0)

                S_db_l = np.flipud(20 * np.log10(S_l))
                S_db_r = np.flipud(20 * np.log10(S_r))
                S_db_d = np.flipud(20 * np.log10(S_d))

                S_db_aux = np.dstack([S_db_l, S_db_r, S_db_d])

                filename_aux = os.path.splitext(os.path.basename(audio_files[ii]))[0]
                if val:
                    with open('../data/audiovisual/audio_spectrograms/'
                              'gammatone_val_{}/{}_{}.pickle'.format(folder_name, filename_aux, jj), 'wb') as f:
                        pickle.dump([S_db_aux, audio_labels[ii]], f)
                    f.close()
                else:
                    with open('../data/audiovisual/audio_spectrograms/'
                              'gammatone_train_{}/{}_{}.pickle'.format(folder_name, filename_aux, jj), 'wb') as f:
                        pickle.dump([S_db_aux, audio_labels[ii]], f)

                    f.close()


def compact_audio_files(path2spectrograms, path2val_spectrograms,
                        n_mels, n_channels, input_name, rep='mel'):
    if os.path.isdir('../data/audiovisual/audio_spectrograms/{}_setup/'.format(input_name)) is False:
        os.mkdir('../data/audiovisual/audio_spectrograms/{}_setup/'.format(input_name))

    onlyfiles = [f for f in listdir(path2spectrograms) if isfile(join(path2spectrograms, f))]

    if n_channels is not None:
        if rep == 'gammatone':
            features = np.zeros((len(onlyfiles), n_mels, 48, n_channels))
        else:
            features = np.zeros((len(onlyfiles), n_mels, 50, n_channels))
    else:
        features = np.zeros((len(onlyfiles), n_mels, 50))
    labels = [None] * len(onlyfiles)

    for i in tqdm(range(0, len(onlyfiles))):
        with open(path2spectrograms + '/' + onlyfiles[i], 'rb') as f:  # Python 3: open(..., 'rb')
            features_aux, labels_aux = pickle.load(f)
        features[i] = features_aux[:, 0:48, :]
        labels[i] = labels_aux

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    label_int = label_encoder.transform(labels)

    # Saving the objects:
    # with open('training_setup.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([features, labels], f)
    if rep == 'gammatone':
        hf = h5py.File('../data/audiovisual/audio_spectrograms/{}_setup/'
                       'training_setup_{}_{}.h5'.format(input_name, input_name, rep), 'w')
    else:
        hf = h5py.File('../data/audiovisual/audio_spectrograms/{}_setup/'
                       'training_setup_{}.h5'.format(input_name, input_name), 'w')
    hf.create_dataset("features", data=features)
    hf.create_dataset("labels", data=label_int)
    hf.close()

    onlyfiles_val = [f for f in listdir(path2val_spectrograms) if isfile(join(path2val_spectrograms, f))]

    if n_channels is not None:
        if rep == 'gammatone':
            features_val = np.zeros((len(onlyfiles_val), n_mels, 48, n_channels))
        else:
            features_val = np.zeros((len(onlyfiles_val), n_mels, 50, n_channels))
    else:
        features_val = np.zeros((len(onlyfiles_val), n_mels, 50))
    labels_val = [None] * len(onlyfiles_val)

    for i in tqdm(range(0, len(onlyfiles_val))):
        with open(path2val_spectrograms + '/' + onlyfiles_val[i], 'rb') as f:  # Python 3: open(..., 'rb')
            features_aux, labels_aux = pickle.load(f)
        features_val[i] = features_aux[:, 0:48, :]
        labels_val[i] = labels_aux

    labels_val_int = label_encoder.transform(labels_val)

    if rep == 'gammatone':
        hf = h5py.File('../data/audiovisual/audio_spectrograms/{}_setup/'
                       'val_setup_{}_{}.h5'.format(input_name, input_name, rep), 'w')
    else:
        hf = h5py.File('../data/audiovisual/audio_spectrograms/{}_setup/'
                       'val_setup_{}.h5'.format(input_name, input_name), 'w')
    hf.create_dataset("features", data=features_val)
    hf.create_dataset("labels", data=labels_val_int)
    hf.close()

    encoder_file = open('../data/audiovisual/audio_spectrograms/{}_setup/'
                        'encoder_{}_{}.pkl'.format(input_name, input_name, rep), 'wb')
    pickle.dump(label_encoder, encoder_file)
    encoder_file.close()


if __name__ == '__main__':
    path2csv_train = '../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/evaluation_setup' \
                     '/fold1_train.csv'
    path2csv_evaluate = '../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/evaluation_setup' \
                        '/fold1_evaluate.csv'

    folder_name = 'hpd'
    get_1s_mel_spectrogram(path2csv_train, mono=False, val=False, folder_name=folder_name)
    get_1s_mel_spectrogram(path2csv_evaluate, mono=False, val=True, folder_name=folder_name)
    path2spectrograms = '../data/audiovisual/audio_spectrograms/train_{}'.format(folder_name)
    path2spectrograms_val = '../data/audiovisual/audio_spectrograms/val_{}'.format(folder_name)
    n_mels = 64
    compact_audio_files(path2spectrograms, path2spectrograms_val,
                        n_mels, 3, folder_name)

    folder_name = 'lrd'
    get_1s_mel_spectrogram(path2csv_train, mono=False, val=False, folder_name=folder_name)
    get_1s_mel_spectrogram(path2csv_evaluate, mono=False, val=True, folder_name=folder_name)
    path2spectrograms = '../data/audiovisual/audio_spectrograms/train_{}'.format(folder_name)
    path2spectrograms_val = '../data/audiovisual/audio_spectrograms/val_{}'.format(folder_name)
    n_mels = 64
    compact_audio_files(path2spectrograms, path2spectrograms_val,
                        n_mels, 3, folder_name)

    folder_name = 'lrd'
    get_1s_gammatone_spectrogram(path2csv_train, mono=False, val=False, folder_name=folder_name)
    get_1s_gammatone_spectrogram(path2csv_evaluate, mono=False, val=True, folder_name=folder_name)
    path2spectrograms = '../data/audiovisual/audio_spectrograms/gammatone_train_{}'.format(folder_name)
    path2spectrograms_val = '../data/audiovisual/audio_spectrograms/gammatone_val_{}'.format(folder_name)
    n_mels = 64
    compact_audio_files(path2spectrograms, path2spectrograms_val,
                       n_mels, 3, folder_name, rep='gammatone')


