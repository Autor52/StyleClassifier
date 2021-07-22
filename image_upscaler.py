import cv2
import os
import pickle
import numpy as np

if __name__ == "__main__":
    file = open('path_list.txt', 'rb')
    path_list = pickle.load(file)
    file.close()

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('EDSR_models\\EDSR_x4.pb')
    sr.setModel("edsr", 4)

    N = len(path_list)

    i = 0
    N100 = np.floor(N/100)
    for path in path_list:
        if i % N100 == 0:
            print("Finished", i/N100, '%')
        i += 1
        image = cv2.imread(path)
        if image.shape[0] < 300 or image.shape[1] < 300:
            image = sr.upsample(image)
            cv2.imwrite(path, image)
