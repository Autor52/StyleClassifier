# test rozmiaru i punktow kluczowych obrazow

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    fast = cv2.FastFeatureDetector_create()
    image_paths = glob.glob("Disabled_classes\\abstract+art\\*")
    avg_sizes = []
    sizes = []
    num_kps = []
    for path in image_paths:
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmp = fast.detect(image_gray, None)
        kp, _ = brief.compute(image_gray, tmp)
        num_kps += [len(kp)]
        sizes.append(image_gray.shape[0]*image_gray.shape[1])

    min_kp = np.min(num_kps)
    avg_kp = round(np.average(num_kps))
    max_kp = np.max(num_kps)
    avg_sizes.append(round(np.average(sizes)))

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle("Number of keypoints in abstract images")
    ax.bar(list(range(3)), [min_kp, avg_kp, max_kp])
    plt.setp(ax, xticks=list(range(3)), xticklabels=['Min', 'Average', 'Max'])
    ax.set_xlabel("Value type")
    ax.set_ylabel("Number of keypoints")
    fig.savefig("checkpoints\\abstract_keypoints.png", format='png', dpi=300)
    print("abstract+art")
    print("min:", min_kp, "avg:", avg_kp, "max:", max_kp)

    data_dir = "verified\\"
    classes = [str.split(ctg, sep="\\")[1] for ctg in glob.glob(data_dir + "\\**")]
    image_paths = []
    for cl in classes:
        image_paths.append(glob.glob(data_dir + cl + "\\*"))

    for cl, paths in zip(classes, image_paths):
        num_kps = []
        sizes = []
        for path in paths:
            image = cv2.imread(path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tmp = fast.detect(image_gray, None)
            kp, _ = brief.compute(image_gray, tmp)
            num_kps += [len(kp)]
            sizes.append(image_gray.shape[0]*image_gray.shape[1])

        min_kp = np.min(num_kps)
        avg_kp = round(np.average(num_kps))
        max_kp = np.max(num_kps)
        avg_sizes.append(round(np.average(sizes)))

        fig, ax = plt.subplots(figsize=(18, 10))
        fig.suptitle("Number of keypoints in " + cl + " images")
        ax.cla()
        ax.bar(list(range(3)), [min_kp, avg_kp, max_kp])
        plt.setp(ax, xticks=list(range(3)), xticklabels=['Min', 'Average', 'Max'])
        ax.set_xlabel("Value type")
        ax.set_ylabel("Number of keypoints")
        fig.savefig("checkpoints\\"+cl+"_keypoints.png", format='png', dpi=300)
        print(cl)
        print("min:", min_kp, "avg:", avg_kp, "max:", max_kp)

    all_classes = ['abstract'] + classes
    fig.suptitle("Average size of images per class")
    ax.cla()
    ax.bar(list(range(5)), avg_sizes)
    plt.setp(ax, xticks=list(range(len(all_classes))), xticklabels=all_classes)
    ax.set_xlabel("Value type")
    ax.set_ylabel("Size [px - height*width]")
    fig.savefig("checkpoints\\Avg_img_sizes.png", format='png', dpi=300)
