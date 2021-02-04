import image_networks
import audio_networks
import config
from keras.layers import concatenate, Dense
from keras.models import Model


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


if __name__ == '__main__':
    construct_joint_network()
