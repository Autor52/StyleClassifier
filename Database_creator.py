# Nieuzywany

import cv2
import pickle
import glob


class ImageDescriptor:
    def __init__(self, image, keypoints, descriptors, filename, query):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.filename = filename
        self.query = query


if __name__ == "__main__":
    data_filename = 'database.txt'
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    fast = cv2.FastFeatureDetector_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    database = []
    try:
        file = open(data_filename, 'rb')
        database = pickle.load(file)
        file.close()
    except FileNotFoundError:
        pass
    except EOFError:
        pass

    dirs = glob.glob("images\\*\\")
    path_list = []
    for d in dirs:
        image_paths = glob.glob(d+'*')
        for img_p in image_paths:
            path_list.append(img_p)

    for path in path_list:
        image = cv2.imread(path)
        grayver = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        initial_kp = fast.detect(grayver, None)
        kp, des = brief.compute(grayver, initial_kp)
        image_in_database = False

        splt = str.split(path, sep='\\')
        query = splt[-2]
        filename = splt[-1]

        for desc in database:
            try:
                matches = bf.match(des, desc.descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                good = []
                for m in matches:
                    if m.distance < 5:
                        good.append(m)
                    else:
                        break
                if len(good) != 0:
                    hit = cv2.drawMatches(desc.image, desc.keypoints, image, kp, good, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow('Match', hit)
                    k = cv2.waitKey(0)
                    if k == ord('1'):
                        image_in_database = True
                        print("Preserving first alternative")
                        break
                    elif k == ord('2'):
                        ind = database.index(desc)
                        database[ind] = ImageDescriptor(image, kp, des, filename, query)
                        image_in_database = True
                        print("Preserving second alternative")
                        break
                    cv2.destroyWindow('Match')
            except cv2.error as e:
                index = path_list.index(path)
                index2 = database.index(desc)
                print(e,
                      "\nat image:", index, '-', filename,
                      '\nvs image from dataset:', index2, '-', desc.filename,
                      '\nskipping action...')
                pass

        if image_in_database:
            continue
        else:
            database.append(ImageDescriptor(image, kp, des, filename, query))

    file = open(data_filename, 'wb')
    pickle.dump(database, file)
    file.close()




