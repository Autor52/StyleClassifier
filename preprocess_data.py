# preprocessing dla wykorzystanych modeli sieci

import numpy as np
import pickle
import glob
import os

from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle

from test_structures import test_structure_basic, test_structure_small, test_structure_large


if __name__ == "__main__":
    data_dir = "verified\\"
    classes = [str.split(ctg, sep="\\")[1] for ctg in glob.glob(data_dir + "\\**")]
    image_paths = []
    for cl in classes:
        image_paths.append(glob.glob(data_dir + cl + "\\*"))

    num_classes = len(classes)
    for test_structure in [test_structure_basic, test_structure_small, test_structure_large]:
        for model_name, definitions in test_structure.items():
            data = None
            category = []

            print("Preprocessing", model_name)
            N = len(image_paths[0])
            N100 = int(np.ceil(N / 100))
            for i in range(N):
                if np.mod(i, N100) == 0:
                    print(i / N100, '%')

                images = []
                paths = []
                for pt in image_paths:
                    paths.append(pt[i])

                for path in paths:
                    j = paths.index(path)
                    img = image.load_img(path, target_size=definitions['pref-size'])
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    if data is None:
                        data = x
                    else:
                        data = np.concatenate((data, x), axis=0)

                    category += [j]

            data = definitions['Preprocessing_function'](data)
            category = np.array(category)
            print("Complete!")

            data, category = shuffle(data, category, random_state=0)

            print("Saving data...")

            data_filepath = "E:\\tf\\data\\" + model_name

            try:
                os.mkdir(data_filepath)
            except FileExistsError:
                pass

            file = open(data_filepath + '\\data', 'wb')
            pickle.dump(data, file)
            file.close()
            file = open(data_filepath + '\\classes', 'wb')
            pickle.dump(category, file)
            file.close()
            file = open(data_filepath + '\\class_names', 'wb')
            pickle.dump(classes, file)
            file.close()
            print("Complete!")
