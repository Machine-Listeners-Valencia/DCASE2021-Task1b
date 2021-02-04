from keras.applications import Xception, InceptionResNetV2, InceptionV3
from keras.layers import Dense
from keras.models import Model
import config


def construct_keras_image_network(include_classification=True, **parameters):
    """
    Constructs a keras model with pretrained weights and top layers according to the classification problem
    Args:
        include_classification (bool): include classification layer
        **parameters: setting use to construct the network
            net (str): pretrained network type (xception, inception_resnet_v2 or inception_v3)
            include_top (bool): if including classification layer, typically set to False
            pooling (str): class of pooling before Dense layers: 'max' or 'avg'
            input_shape (tuple): size of the RBG images, typically (224, 224, 3)
            trainable (bool): if pretrained network layers can be optimized
            verbose (bool): if summary is showed

    Returns:
        keras model according to the type and the Dense layers at the top
    """

    net = parameters['net']
    input_shape = parameters['input_shape']
    include_top = parameters['include_top']
    pooling = parameters['pooling']
    trainable = parameters['trainable']
    verbose = parameters['verbose']

    if net is 'xception':
        base_model = Xception(input_shape=input_shape, include_top=include_top, pooling=pooling)

    elif net is 'inception_resnet_v2':
        base_model = InceptionResNetV2(input_shape=input_shape, include_top=include_top, pooling=pooling)

    elif net is 'inception_v3':
        base_model = InceptionV3(input_shape=input_shape, include_top=include_top, pooling=pooling)

    for layer in base_model.layers:
        layer.trainable = trainable

    if include_classification:
        outputs = Dense(units=config.n_classes, activation='softmax')(base_model.layers[-1].output)
    else:
        outputs = base_model.layers[-1].output

    model = Model(inputs=base_model.inputs, outputs=outputs)

    if verbose:
        print(model.summary())

    return model


if __name__ == '__main__':
    model = construct_keras_image_network(include_classification=True,
                                          **config.image_network_settings)
