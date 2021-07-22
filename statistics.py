# skrypt sprawdzajacy wytrenowane modele

import glob
import pickle
import multiprocessing

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from test_structures import test_structure_basic
from tensorflow.keras.preprocessing import image


def do_statistics(split_folder, split_folders, data, category, classes, fig, ax, model_name):
    # tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model(split_folder + "\\best_model.hdf5")
    test_indices = pickle.load(open(split_folder + "\\test_indices", 'rb'))

    class_accuracy = []

    data_test, category_test = data[test_indices], category[test_indices]
    outconf = model.predict(data_test)
    guessed_classes = np.array([list(p).index(np.max(p)) for p in outconf])
    for cl in classes:
        id = classes.index(cl)
        tmp1 = category_test == id
        tmp2 = guessed_classes == id
        correct_guesses_class = np.logical_and(tmp1, tmp2)

        class_accuracy.append(np.sum(correct_guesses_class) / np.sum(tmp1))

        class_confidences = outconf[correct_guesses_class]
        class_confidence_values = []
        for confs in class_confidences:
            class_confidence_values += [np.max(confs)]

        min_class_conf = np.min(class_confidence_values)
        avg_class_conf = np.average(class_confidence_values)
        max_class_conf = np.max(class_confidence_values)
        conf_list = [min_class_conf, avg_class_conf, max_class_conf]

        ax.cla()
        fig.suptitle(model_name + " " + cl + " confidence values, split: " + str(split_folders.index(split_folder)))
        ax.bar(list(range(3)), conf_list)
        plt.setp(ax, xticks=list(range(3)), xticklabels=['Minimum', 'Average', 'Maximum'])
        ax.set_xlabel("Confidence value type")
        ax.set_ylabel("Confidence value")
        for i in range(3):
            ax.text(i,
                    conf_list[i] / 2,
                    str(round(conf_list[i], 2)),
                    fontweight='bold')
        fig.savefig(split_folder + "\\confidence_statistics_" + cl + ".png", format='png', dpi=300)

    ax.cla()
    fig.suptitle(model_name + " accuracy per class, split: " + str(split_folders.index(split_folder)))
    ax.bar(list(range(len(classes))), class_accuracy)
    plt.setp(ax, xticks=list(range(len(classes))), xticklabels=classes)
    ax.set_xlabel('Art style')
    ax.set_ylabel('Accuracy')
    for i in range(len(classes)):
        ax.text(i,
                class_accuracy[i] / 2,
                str(round(class_accuracy[i], 2)),
                fontweight='bold')
    fig.savefig(split_folder + "\\accuracy_statistics.png", format='png', dpi=300)

    definitions = test_structure_basic[model_name]
    abstract_paths = glob.glob("Disabled_classes\\abstract+art\\*")
    image_data = []
    for pa in abstract_paths:
        img = image.load_img(pa, target_size=definitions['pref-size'])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if data is None:
            image_data = x
        else:
            image_data = np.concatenate((data, x), axis=0)

    # wyniki klasyfikacji abstraktu
    outabstract = model.predict(image_data)
    guessed_classes_abstract = np.array([list(p).index(np.max(p)) for p in outabstract])
    guessed_classes_abstract_sum = [0, 0, 0, 0]
    for pcl in guessed_classes_abstract:
        guessed_classes_abstract_sum[pcl] += 1

    ax.cla()
    fig.suptitle(model_name + " abstract classification results, split: " + str(split_folders.index(split_folder)))
    ax.bar(list(range(len(classes))), guessed_classes_abstract_sum)
    plt.setp(ax, xticks=list(range(len(classes))), xticklabels=classes)
    ax.set_xlabel('Art style')
    ax.set_ylabel('Number of images')
    for i in range(len(classes)):
        ax.text(i,
                guessed_classes_abstract_sum[i] / 2,
                str(guessed_classes_abstract_sum[i]),
                fontweight='bold')
    fig.savefig(split_folder + "\\abstract_classification.png", format='png', dpi=300)

    # pewnosc modelu co do klasyfikacji
    guessed_classes_confidences = np.array([np.max(p) for p in outabstract])
    min_abs_conf = np.min(guessed_classes_confidences)
    avg_abs_conf = np.average(guessed_classes_confidences)
    max_abs_conf = np.max(guessed_classes_confidences)
    conf_list_abs = [min_abs_conf, avg_abs_conf, max_abs_conf]

    ax.cla()
    fig.suptitle(model_name + " abstract confidence values, split: " + str(split_folders.index(split_folder)))
    ax.bar(list(range(3)), conf_list_abs)
    plt.setp(ax, xticks=list(range(3)), xticklabels=['Minimum', 'Average', 'Maximum'])
    ax.set_xlabel("Confidence value type")
    ax.set_ylabel("Confidence value")
    for i in range(3):
        ax.text(i,
                conf_list_abs[i] / 2,
                str(round(conf_list_abs[i], 2)),
                fontweight='bold')
    fig.savefig(split_folder + "\\confidence_statistics_abstract.png", format='png', dpi=300)


if __name__ == "__main__":
    data_dir = "verified\\"
    path = "E:\\tf\\checkpoints\\"
    folders = glob.glob(path+"**")
    fig, ax = plt.subplots(figsize=(18, 10))
    for folder in folders:
        split_folders = glob.glob(folder+"\\**")
        model_name = str.split(folder, sep='-')[3]
        if model_name == 'sequential':
            model_name += '-'+str.split(folder, sep='-')[4]
        data_filepath = "E:\\tf\\data\\" + model_name
        data = pickle.load(open(data_filepath + "\\data", 'rb'))
        category = pickle.load(open(data_filepath + "\\classes", 'rb'))
        classes = pickle.load(open(data_filepath + "\\class_names", 'rb'))

        for split_folder in split_folders:
            process = multiprocessing.Process(target=do_statistics, args=(split_folder,
                                                                          split_folders,
                                                                          data,
                                                                          category,
                                                                          classes,
                                                                          fig, ax,
                                                                          model_name))
            process.start()
            process.join()



