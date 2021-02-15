import config


def train():
    if config.which_train == 'image':
        from trainers import image_trainer
        from image_networks import construct_keras_image_network
        model = construct_keras_image_network(include_classification=True,
                                              **config.image_network_settings)


        image_trainer(model, config.path2image_data)
    # TODO: import model


if __name__ == '__main__':
    train()
