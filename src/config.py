"""
Configuration file: commented in list form are indicated the supported options
"""
import os

home = os.getenv('HOME')

path2image_data = os.path.join(home, 'repos/DCASE2021-Task1b/data/mit67/indoorCVPR_09/split')
path2audio_data = os.path.join(home, 'repos/DCASE2021-Task1b/data/audiovisual/audio_spectrograms/lrd_setup/')
path2outputs = os.path.join(home, 'repos/DCASE2021-Task1b/training_outputs')
n_classes = 67  # [10]
batch_size = 2
epochs = 150
which_train = 'audio'  # ['image', 'audio']
image_network_settings = {
    # possible nets: [efficientnet-1-2-3-4-5-6-7, inception_resnet_v2, xception, inception_v3,
    # places365 ]
    'net': 'efficientnet-0',
    'include_top': False,
    'pooling': 'avg',
    'input_shape': (224, 224, 3),
    'trainable': True,
    'verbose': True
}
audio_network_settings = {
    'nfilters': (32, 64, 128),
    'pooling': [(1, 10), (1, 5), (1, 5)],
    'dropout': [0.3, 0.3, 0.3],
    'top_flatten': 'avg',
    'ratio': 2,
    'pre_act': False,
    'spectrogram_dim': (64, 500, 3),
    'verbose': True
}

image_train_gen_args = dict(rescale=1. / 255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest'
                            )
image_val_gen_args = dict(rescale=1. / 255)

audio_train_gen_args = {
    'alpha': 0.4
}

callbacks_settings = {
    'monitor': 'val_categorical_accuracy',
    'lr_factor': 0.5,
    'lr_patience': 20,
    'finish_patience': 40
}
