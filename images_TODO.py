# tworzenie listy dla cleanup.py

import glob
import pickle


if __name__ == "__main__":
    dirs = glob.glob("images\\*\\")
    path_list = []
    for d in dirs:
        image_paths = glob.glob(d+'*')
        for img_p in image_paths:
            path_list.append(img_p)
    file = open('path_list.txt', 'wb')
    pickle.dump(path_list, file)
    file.close()
