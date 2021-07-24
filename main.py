# skrypt trenujacy modele

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os
import multiprocessing

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras import mixed_precision

from test_structures import test_structure_basic, test_structure_small, test_structure_large, \
    test_structure_basic_medium


def test_model(model_name, definitions, data_train, data_test, category_train, category_test, split_filepath,
               num_classes):
    tf.config.set_visible_devices([], 'GPU')  # zakomentowac w przypadku korzystania z GPU
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = definitions['Model_definition'](weights=None,
                                            classes=num_classes)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=split_filepath + "\\best_model.hdf5",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # model.summary()
    epochs = 15

    history = model.fit(
        x=data_train,
        y=category_train,
        # batch_size=12,
        validation_data=(data_test, category_test),
        epochs=epochs,
        callbacks=model_checkpoint_callback,
        shuffle=False,
        # steps_per_epoch=60,
        use_multiprocessing=True
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    max_acc = np.max(val_acc)
    max_id = val_acc.index(max_acc)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.scatter([max_id], [max_acc], marker='*')
    plt.annotate(text=str(np.format_float_positional(max_acc, precision=3)), xy=(max_id, max_acc))
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(split_filepath + '\\results-' + model_name + '.png', dpi=300)


if __name__ == "__main__":
    # policy = mixed_precision.Policy('float32')
    # mixed_precision.set_global_policy(policy)

    num_classes = 4
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    # structures = [test_structure_basic, test_structure_small, test_structure_large]
    structures = [test_structure_basic]
    # folders = ['data_abstract', 'checkpoints_abstract']
    folders = ['data', 'checkpoints']

    for test_structure in structures:
        for model_name, definitions in test_structure.items():
            data_filepath = "E:\\tf\\" + folders[0] + "\\" + model_name
            data = pickle.load(open(data_filepath + "\\data", 'rb'))
            category = pickle.load(open(data_filepath + "\\classes", 'rb'))
            classes = pickle.load(open(data_filepath + "\\class_names", 'rb'))

            optimizer = 'adam'

            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H;%M;%S")
            checkpoint_filepath = "E:\\tf\\" + folders[1] + "\\" + dt_string + "-" + model_name + "-" + optimizer

            try:
                os.mkdir(checkpoint_filepath)
            except FileExistsError:
                pass

            split_id = 1

            for train_index, test_index in skf.split(data, category):
                split_filepath = checkpoint_filepath + "\\" + str(split_id)
                try:
                    os.mkdir(split_filepath)
                except FileExistsError:
                    pass

                file = open(split_filepath + '\\train_indices', 'wb')
                pickle.dump(train_index, file)
                file.close()

                file = open(split_filepath + '\\test_indices', 'wb')
                pickle.dump(test_index, file)
                file.close()

                data_train, data_test = data[train_index], data[test_index]
                category_train, category_test = category[train_index], category[test_index]
                # zwalnianie pamieci poprzez kasowanie procesu (glownie dla kart graficznych)
                process = multiprocessing.Process(target=test_model, args=(model_name,
                                                                           definitions,
                                                                           data_train,
                                                                           data_test,
                                                                           category_train,
                                                                           category_test,
                                                                           split_filepath,
                                                                           num_classes))
                process.start()
                process.join()

                split_id += 1
