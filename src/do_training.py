import config


def train():
    if config.which_train == 'image':
        from trainers import image_trainer
    # TODO: import model


if __name__ == '__main__':
    train()
