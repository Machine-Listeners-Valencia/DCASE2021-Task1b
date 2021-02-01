"""
Configuration file: commented in list form are indicated the supported options
"""

path2image = ''
pre_trained_network = 'xception'  # ['xception', 'inception_resnet_v2', 'inception_v3']
n_classes = 10  # [10]
which_train = 'image'  # ['image', 'audio']
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
