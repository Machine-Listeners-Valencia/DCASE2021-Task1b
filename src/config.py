"""
Configuration file: commented in list form are indicated the supported options
"""

path2image = ''
pre_trained_network = 'xception'  # ['xception', 'inception_resnet_v2', 'inception_v3']
n_classes = 10  # [10]
batch_size = 32
epochs = 250
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

image_network_setting= {
    'target_size': (150, 150)
}
image_train_gen_args = dict(rescale=1./255)
image_val_gen_args = dict(rescale=1./255)