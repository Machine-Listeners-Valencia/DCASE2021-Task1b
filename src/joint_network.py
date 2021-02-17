import image_networks
import audio_networks
import config
import tensorflow
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.models import Model


def construct_joint_network(verbose=True):
    image_model = image_networks.construct_keras_image_network(False,
                                                               **config.image_network_settings)
    audio_model = audio_networks.construct_asc_network_csse(False,
                                                            **config.audio_network_settings)

    # TODO: merge models
    #  https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0

    model_concat = concatenate([image_model.output, audio_model.output], axis=-1)
    model_concat = Dense(config.n_classes, activation='softmax')(model_concat)
    model = Model(inputs=[image_model.input, audio_model.input], outputs=model_concat)

    if verbose:
        print(model.summary())

    return model


def construct_dummy_joint_network(verbose=True):
    """
    Dummy joint network that can be tested in RAMs with low capacity. JUST FOR TESTING
    Args:
        verbose ():

    Returns:

    """
    audio_model = audio_networks.construct_asc_network_csse(False,
                                                            **config.audio_network_settings)

    image_model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.Input(shape=config.image_network_settings['input_shape']),
            tensorflow.keras.layers.Conv2D(32, 5, strides=2, activation="relu"),
            tensorflow.keras.layers.GlobalAveragePooling2D(),
        ]
    )

    model_concat = concatenate([image_model.output, audio_model.output], axis=-1)
    model_concat = Dense(config.n_classes, activation='softmax')(model_concat)
    model = Model(inputs=[image_model.input, audio_model.input], outputs=model_concat)

    if verbose:
        print(model.summary())

    return model


if __name__ == '__main__':
    construct_joint_network()
