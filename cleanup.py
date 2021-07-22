# Wycinanie obrazow ze zdjec

import cv2
import os
import pickle


if __name__ == "__main__":
    file = open('path_list.txt', 'rb')
    path_list = pickle.load(file)
    file.close()

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    to_remove = []
    print("Images left:", len(path_list))

    for path in path_list:
        splt = str.split(path, sep='\\')
        query = splt[-2]
        filename = str.split(splt[-1], sep='.')[0]
        frmt = str.split(splt[-1], sep='.')[1]

        print("Image category:", query)
        image = cv2.imread(path)

        i = 0
        while True:
            try:
                r = cv2.selectROI('Display', image)
            except cv2.error as e:
                to_remove.append(path)
                print(e, '\nremoving path...')
                break
            if r == (0, 0, 0, 0):
                break
            else:
                try:
                    imCrop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                    new_name = filename + str(i) + '.' + frmt
                    new_path = 'images\\' + query + '\\' + new_name
                    cv2.imwrite(new_path, imCrop)
                    image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = [0, 0, 0]
                    i += 1
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(e, '\ndumping data...')
                    for p in to_remove:
                        try:
                            path_list.remove(p)
                        except ValueError:
                            pass
                    file = open('path_list.txt', 'wb')
                    pickle.dump(path_list, file)
                    file.close()
                    exit(1)

        to_remove.append(path)
        print('Press e to stop, b to remove image or other keys to continue')
        k = cv2.waitKey(0)
        if k == ord('e'):
            break
        elif k == ord('b'):
            if os.path.exists(path):
                os.remove(path)

    for path in to_remove:
        try:
            path_list.remove(path)
        except ValueError as e:
            print(e, '\nskipping action...')
            pass

    file = open('path_list.txt', 'wb')
    pickle.dump(path_list, file)
    file.close()

    print("Images left:", len(path_list))


