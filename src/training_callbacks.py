from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# TODO: check options

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
