# TODO: add mixup, image plus audio

import numpy as np
import tensorflow


class AudioMixupGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, y_train, batch_size=32, alpha=0.2, shuffle=True):
        'Initialization'
        self.X_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(x_train)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sample_num / (self.batch_size * 2)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size * 2:(index + 1) * self.batch_size * 2]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.sample_num)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

if __name__ == '__main__':

    import config

    test_audio = True

    if test_audio:
        from audio_networks import construct_asc_network_csse

        dummy_audio_features = np.random.rand(2000, config.audio_network_settings['spectrogram_dim'][0],
                                              config.audio_network_settings['spectrogram_dim'][1],
                                              config.audio_network_settings['spectrogram_dim'][2])
        dummy_labels = np.random.rand(2000, config.n_classes)

        model = construct_asc_network_csse(**config.audio_network_settings)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        train_datagen = AudioMixupGenerator(dummy_audio_features, dummy_labels)

        model.fit(train_datagen)

