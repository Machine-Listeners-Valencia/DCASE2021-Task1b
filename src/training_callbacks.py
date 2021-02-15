from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import config


def create_csv_logger(path2store):
    logger = CSVLogger(path2store)

    return logger


def create_early_stopping(monitor, patience):
    early_stopping = EarlyStopping(monitor=monitor, patience=patience)

    return early_stopping


def create_reduce_lr(monitor, factor, patience):
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience)

    return reduce_lr


def create_checkpoint(path2model, monitor, save_best=True):
    checkpoint = ModelCheckpoint(path2model, monitor=monitor, save_best_only=save_best)

    return checkpoint


def create_callbacks(path2store, **parameters):
    monitor = parameters['monitor']
    lr_factor = parameters['lr_factor']
    lr_patience = parameters['lr_patience']
    finish_patience = parameters['finish_patience']

    best_checkpoint = create_checkpoint(os.path.join(path2store, 'best.h5'), monitor, save_best=True)
    last_checkpoint = create_checkpoint(os.path.join(path2store, 'last.h5'), monitor, save_best=False)
    reduce_lr = create_reduce_lr(monitor, lr_factor, lr_patience)
    early_stopping = create_early_stopping(monitor, finish_patience)
    csv_logger = create_csv_logger(os.path.join(path2store, 'logger.csv'))

    return [best_checkpoint, last_checkpoint, reduce_lr, early_stopping, csv_logger]


if __name__ == '__main__':
    home = os.getenv('HOME')

    callbacks = create_callbacks(os.path.join(home, '/repos/DCASE2021-Task1b/data/dummy_test'),
                                 **config.callbacks_settings)
