# skrypt wizualizujacy modele
import glob
import tensorflow as tf

from tensorflow.keras.utils import plot_model

if __name__ == "__main__":
    path = "E:\\tf\\checkpoints\\"
    folders = glob.glob(path+"**")
    tf.config.set_visible_devices([], 'GPU')
    for folder in folders:
        split_folders = glob.glob(folder+"\\**")
        model_name = str.split(folder, sep='-')[3]
        if model_name == 'sequential':
            model_name += '-'+str.split(folder, sep='-')[4]

        for split_folder in split_folders:
            model = tf.keras.models.load_model(split_folder + "\\best_model.hdf5")
            plot_model(model, to_file=split_folder+"\\model_vis.png", show_shapes=True)
