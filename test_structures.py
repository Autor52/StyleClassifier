# slowniki upraszczajace testowanie wielu modeli

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, MobileNet, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.applications import Xception, ResNet152V2, InceptionResNetV2, DenseNet201


def basic_preprocess(x):
    return np.divide(x, 255)


def basic_4(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


def basic_5(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


def basic_6(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


def basic_7(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


def basic_8(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


def basic_15(weights=None, classes=5, classifier_activation='softmax'):
    model = Sequential([
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(classes, activation=classifier_activation)
    ])
    return model


test_structure_small = {
    "VGG16": {
        "Preprocessing_function": tf.keras.applications.vgg16.preprocess_input,
        "Model_definition": VGG16,
        "pref-size": (224, 224)
    },
    "MobileNet": {
        "Preprocessing_function": tf.keras.applications.mobilenet.preprocess_input,
        "Model_definition": MobileNet,
        "pref-size": (224, 224)
    },
    "MobileNetV2": {
        "Preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input,
        "Model_definition": MobileNetV2,
        "pref-size": (224, 224)
    },
    "NASNetMobile": {
        "Preprocessing_function": tf.keras.applications.nasnet.preprocess_input,
        "Model_definition": NASNetMobile,
        "pref-size": (224, 224)
    },
    "EfficientNetB0": {
        "Preprocessing_function": tf.keras.applications.efficientnet.preprocess_input,
        "Model_definition": EfficientNetB0,
        "pref-size": (224, 224)
    }
}

test_structure_large = {
    "Xception": {
        "Preprocessing_function": tf.keras.applications.xception.preprocess_input,
        "Model_definition": Xception,
        "pref-size": (299, 299)
    },
    "ResNet152V2": {
        "Preprocessing_function": tf.keras.applications.resnet_v2.preprocess_input,
        "Model_definition": ResNet152V2,
        "pref-size": (224, 224)
    },
    "InceptionResNetV2": {
        "Preprocessing_function": tf.keras.applications.inception_resnet_v2.preprocess_input,
        "Model_definition": InceptionResNetV2,
        "pref-size": (299, 299)
    },
    "DenseNet201": {
        "Preprocessing_function": tf.keras.applications.densenet.preprocess_input,
        "Model_definition": DenseNet201,
        "pref-size": (224, 224)
    }
}

test_structure_basic = {
    "sequential-4": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_4,
        "pref-size": (299, 299)
    },
    "sequential-5": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_5,
        "pref-size": (299, 299)
    },
    "sequential-6": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_6,
        "pref-size": (299, 299)
    },
    "sequential-7": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_7,
        "pref-size": (299, 299)
    },
    "sequential-8": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_8,
        "pref-size": (299, 299)
    }
}

test_structure_basic_medium = {
    "sequential-15": {
        "Preprocessing_function": basic_preprocess,
        "Model_definition": basic_15,
        "pref-size": (299, 299)
    }
}
